#include "nnet/nnet-trnopts.h"
#include "nnet/nnet-nnet.h"
#include "nnet/nnet-loss.h"
#include "nnet/nnet-randomizer.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "base/timer.h"
#include "cudamatrix/cu-device.h"

#include "util.h"
#include "nnet-my-nnet.h"
#include <sstream>
#include <omp.h>

using namespace std;
using namespace kaldi;
using namespace kaldi::nnet1;

int main(int argc, char *argv[]) {

   try {
      string usage;
      usage.append("Perform one iteration of My Neural Network LSTM training by Stochastic Gradient Descent, shuffling in sentence level. \n")
         .append("Use feature, label and path to train the neural net. \n")
         .append("Usage: ").append(argv[0]).append(" [options] <feature-rspecifier> <label-rspecifier> <score-path-rspecifier> <nnet-in> [<nnet-out>]\n")
         .append("e.g.: \n")
         .append(" ").append(argv[0]).append(" ark:feat.ark ark:lab.ark \"ark:lattice-to-vec ark:1.lat ark:- |\" nnet.init nnet.iter1\n");

      ParseOptions po(usage.c_str());

      NnetTrainOptions trn_opts;
      trn_opts.Register(&po);

      bool binary = true, 
           crossvalidate = false;

      po.Register("binary", &binary, "Write output in binary mode");
      po.Register("cross-validate", &crossvalidate, "Perform cross-validation (don't backpropagate)");

      string use_gpu="yes";
      po.Register("use-gpu", &use_gpu, "yes|no|optional, only has effect if compiled with CUDA"); 

      int negative_num = 0;
      po.Register("negative-num", &negative_num, "insert negative example in training");

      string feature_transform;
      po.Register("feature-transform", &feature_transform, "Feature transform in Nnet format");

      int32 targets_delay=5;
      po.Register("targets-delay", &targets_delay, "---LSTM--- BPTT targets delay"); 

      int32 batch_size=64;
      po.Register("batch-size", &batch_size, "---LSTM--- BPTT batch size"); 

      int32 num_stream=8;
      po.Register("num-stream", &num_stream, "---LSTM--- BPTT multi-stream training"); 

      int32 dump_interval=0;
      po.Register("dump-interval", &dump_interval, "---LSTM--- num utts between model dumping [ 0 == disabled ]"); 

      bool randomize = false;
      NnetDataRandomizerOptions rnd_opts;
      rnd_opts.Register(&po);
      po.Register("randomize", &randomize, "dummy options...");

      po.Read(argc, argv);

      if (po.NumArgs() != 5-(crossvalidate?1:0)) {
         po.PrintUsage();
         exit(1);
      }

      string feat_rspecifier       = po.GetArg(1),
             label_rspecifier      = po.GetArg(2),
             score_path_rspecifier = po.GetArg(3),
             nnet_in_filename      = po.GetArg(4),
             nnet_out_filename;

      if(!crossvalidate){
         nnet_out_filename = po.GetArg(5);
      }

      //Select the GPU
#if HAVE_CUDA==1
      LockSleep(GPU_FILE);
      CuDevice::Instantiate().SelectGpuId(use_gpu);
#endif

      Nnet nnet_transf;
      if(feature_transform != "") {
         nnet_transf.Read(feature_transform);
      }

      MyNnet nnet;
      nnet.Read(nnet_in_filename);
      nnet.SetTrainOptions(trn_opts);

      int labelNum = nnet.GetLabelNum();

      Timer time;
      KALDI_LOG << (crossvalidate?"CROSS-VALIDATION":"TRAINING") << " STARTED";

      // ------------------------------------------------------------
      // read in all features and save to CPU (for shuffling)
      // read in all labels and save to CPU (for shuffling)
      SequentialScorePathReader         score_path_reader(score_path_rspecifier);
      SequentialBaseFloatMatrixReader   feature_reader(feat_rspecifier);
      SequentialUcharVectorReader       label_reader(label_rspecifier);

      int64 total_frames = 0;
      int32 num_done = 0;

      Xent xent;

      int seqs_stride = 1 + score_path_reader.Value().Value().size() + negative_num;

      //  book-keeping for multi-streams
      vector< string >            keys(num_stream);
      vector< Matrix<BaseFloat> > feats(num_stream);
      vector< vector< uchar > >   labels(num_stream); 
      vector< vector< uchar > >   targets(num_stream);

      vector<int> curt(num_stream, 0);
      vector<int> lent(num_stream, 0);
      vector<int> new_utt_flags(num_stream, 0);

      // bptt batch buffer
      int32 feat_dim = nnet.InputDim();

      CuMatrix<BaseFloat> feat_transf, nnet_out, obj_diff;

      Vector<BaseFloat>   frame_mask(batch_size * num_stream * seqs_stride, kSetZero);
      vector<int32>       lab(batch_size * num_stream * seqs_stride);
      Posterior           target(batch_size * num_stream * seqs_stride);

      Matrix<BaseFloat>   feat(batch_size * num_stream, feat_dim, kSetZero);

      srand(rnd_opts.randomizer_seed);

      vector< pair<int32, BaseFloat> > crr;
      crr.push_back(make_pair(0, 1.0));

      vector< pair<int32, BaseFloat> > incrr;
      incrr.push_back(make_pair(1, 1.0));

      while (1) {
         // loop over all streams, check if any stream reaches the end of its utterance,
         // if any, feed the exhausted stream with a new utterance, update book-keeping infos
         for (int s = 0; s < num_stream; s++) {
            // this stream still has valid frames
            if (curt[s] < lent[s]) {
               new_utt_flags[s] = 0;
               continue;
            }
            // else, this stream exhausted, need new utterance
            while (!(feature_reader.Done() || score_path_reader.Done() || label_reader.Done())) {
               assert( score_path_reader.Key() == feature_reader.Key() );
               assert( label_reader.Key() == feature_reader.Key() );

               const ScorePath::Table  &table = score_path_reader.Value().Value();
               const Matrix<BaseFloat> &mat   = feature_reader.Value();
               const vector<uchar>     &label = label_reader.Value();

               assert(seqs_stride == 1 + table.size() + negative_num);

               // forward the features through a feature-transform,
               nnet_transf.Feedforward(CuMatrix<BaseFloat>(mat), &feat_transf);

               vector< uchar > &seq = labels[s];
               vector< uchar > &tgt = targets[s];

               seq.resize( label.size() * seqs_stride );
               for(int j = 0; j < label.size(); ++j)
                  seq[j * seqs_stride] = label[j];

               for(int i = 0; i < table.size(); ++i)
                  for(int j = 0; j < label.size(); ++j)
                     seq[j * seqs_stride + i + 1] = table[i].second[j];

               for(int i = 0; i < negative_num; ++i)
                  for(int j = 0; j < label.size(); ++j)
                     seq[j * seqs_stride + i + 1 + table.size()] = rand()%labelNum + 1;

               tgt.resize(seq.size());
               for(int i = 0; i < label.size(); ++i)
                  for(int j = 0; j < seqs_stride; ++j)
                     tgt[i * seqs_stride + j] =
                        (seq[i*seqs_stride + j] == label[i]) ? 1:0;

               // checks ok, put the data in the buffers,
               keys[s] = feature_reader.Key();
               feats[s].Resize(feat_transf.NumRows(), feat_transf.NumCols());
               feat_transf.CopyToMat(&feats[s]); 
               //targets[s] = target;
               curt[s] = 0;
               lent[s] = feats[s].NumRows();
               new_utt_flags[s] = 1;  // a new utterance feeded to this stream

               feature_reader.Next();
               score_path_reader.Next();
               label_reader.Next();
               break;
            }
         }

         // we are done if all streams are exhausted
         int done = 1;
         for (int s = 0; s < num_stream; s++) {
            if (curt[s] < lent[s]) done = 0;  // this stream still contains valid data, not exhausted
         }
         if (done) break;

         // fill a multi-stream bptt batch
         // * frame_mask: 0 indicates padded frames, 1 indicates valid frames
         // * target: padded to batch_size
         // * feat: first shifted to achieve targets delay; then padded to batch_size
         for (int t = 0; t < batch_size; t++) {
            for (int s = 0; s < num_stream; s++) {
               int row     = t * num_stream + s;
               int row_dly = curt[s] + targets_delay;

               // frame_mask & targets padding
               for(int i = 0; i < seqs_stride; ++i){
                  if (curt[s] < lent[s]) {
                     frame_mask(row*seqs_stride + i) = 1;
                     target[row*seqs_stride + i] = 
                        targets[s][curt[s]*seqs_stride + i] ? crr: incrr;

                  } else {
                     frame_mask(row*seqs_stride + i) = 0;
                     target[row*seqs_stride + i] = 
                        targets[s][(lent[s] - 1)*seqs_stride + i] ? crr: incrr;
                  }

                  if (row_dly < lent[s]) {
                     lab[row * seqs_stride + i] = labels[s][row_dly * seqs_stride + i] - 1;
                  } else {
                     lab[row * seqs_stride + i] = labels[s][(lent[s] - 1) * seqs_stride + i] -1;
                  }
               }

               // feat shifting & padding
               if (row_dly < lent[s]) {
                  feat.Row(row).CopyFromVec(feats[s].Row(row_dly));
               } else {
                  feat.Row(row).CopyFromVec(feats[s].Row(lent[s]-1));
               }

               curt[s]++;
            }
         }

         nnet.SetLabelSeqs(lab, seqs_stride);
         nnet.ResetLstmStreams(new_utt_flags);

         // forward pass
         nnet.Propagate(CuMatrix<BaseFloat>(feat), &nnet_out);

         xent.Eval(frame_mask, nnet_out, target, &obj_diff);

         // backward pass
         if (!crossvalidate) {
            nnet.Backpropagate(obj_diff, NULL);
         }

         // 1st minibatch : show what happens in network 
         if (kaldi::g_kaldi_verbose_level >= 1 && total_frames == 0) { // vlog-1
            KALDI_VLOG(1) << "### After " << total_frames << " frames,";
            KALDI_VLOG(1) << nnet.InfoPropagate();
            if (!crossvalidate) {
               KALDI_VLOG(1) << nnet.InfoBackPropagate();
               KALDI_VLOG(1) << nnet.InfoGradient();
            }
         }

         int frame_progress = frame_mask.Sum();
         total_frames += frame_progress;

         int num_done_progress = 0;
         for (int i =0; i < new_utt_flags.size(); i++) {
            num_done_progress += new_utt_flags[i];
         }
         num_done += num_done_progress;

         // monitor the NN training
         if (kaldi::g_kaldi_verbose_level >= 2) { // vlog-2
            if ((total_frames-frame_progress)/25000 != (total_frames/25000)) { // print every 25k frames
               KALDI_VLOG(2) << "### After " << total_frames << " frames,";
               KALDI_VLOG(2) << nnet.InfoPropagate();
               if (!crossvalidate) {
                  KALDI_VLOG(2) << nnet.InfoBackPropagate();
                  KALDI_VLOG(2) << nnet.InfoGradient();
               }
            }
         }

         // report the speed
         if ((num_done-num_done_progress)/1000 != (num_done/1000)) {
            double time_now = time.Elapsed();
            KALDI_VLOG(1) << "After " << num_done << " utterances: time elapsed = "
               << time_now/60 << " min; processed " << total_frames/time_now
               << " frames per second.";

#if HAVE_CUDA==1
            // check the GPU is not overheated
            CuDevice::Instantiate().CheckGpuHealth();
#endif
         }

         if (dump_interval > 0) { // disabled by 'dump_interval == 0',
            if ((num_done-num_done_progress)/dump_interval != (num_done/dump_interval)) {
               char nnet_name[512];
               if (!crossvalidate) {
                  sprintf(nnet_name, "%s_utt%d", nnet_out_filename.c_str(), num_done);
                  nnet.Write(nnet_name, binary);
               }
            }
         }
      }

      // after last minibatch : show what happens in network 
      if (kaldi::g_kaldi_verbose_level >= 1) { // vlog-1
         KALDI_VLOG(1) << "### After " << total_frames << " frames,";
         KALDI_VLOG(1) << nnet.InfoPropagate();
         if (!crossvalidate) {
            KALDI_VLOG(1) << nnet.InfoBackPropagate();
            KALDI_VLOG(1) << nnet.InfoGradient();
         }
      }

      if (!crossvalidate) {
         nnet.Write(nnet_out_filename, binary);
      }

      KALDI_LOG << "Done " << num_done << " files, "
         << "[" << (crossvalidate?"CROSS-VALIDATION":"TRAINING")
         << ", " << (randomize?"RANDOMIZED":"NOT-RANDOMIZED") 
         << ", " << time.Elapsed()/60 << " min, fps" << total_frames/time.Elapsed()
         << "]";  

      KALDI_LOG << xent.Report();

#if HAVE_CUDA==1
      CuDevice::Instantiate().PrintProfile();
#endif

      return 0;
   } catch(const exception &e) {
      cerr << e.what();
    return -1;
  }
}
