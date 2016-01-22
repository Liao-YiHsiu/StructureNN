#include "nnet/nnet-trnopts.h"
#include "nnet/nnet-nnet.h"
#include "nnet/nnet-loss.h"
#include "nnet/nnet-randomizer.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "base/timer.h"
#include "cudamatrix/cu-device.h"

#include "my-nnet/nnet-my-nnet.h"
#include "my-nnet/nnet-my-loss.h"
#include "score-path/score-path.h"
#include <sstream>
#include <omp.h>

using namespace std;
using namespace kaldi;
using namespace kaldi::nnet1;

int main(int argc, char *argv[]) {

   try {
      string usage;
      usage.append("Perform one iteration of My Neural Network Training in sentence level (without shuffling.) \n")
         .append("Use feature, label and path to train the neural net. \n")
         .append("Usage: ").append(argv[0]).append(" [options] <feature-rspecifier> <label-rspecifier> <score-path-rspecifier> <loss-func-file> <nnet-in> [<nnet-out>]\n")
         .append("e.g.: \n")
         .append(" ").append(argv[0]).append(" ark:feat.ark ark:lab.ark \"ark:lattice-to-vec ark:1.lat ark:- |\" loss.conf nnet.init nnet.iter1\n");

      ParseOptions po(usage.c_str());

      NnetTrainOptions trn_opts;
      trn_opts.Register(&po);

      bool binary = true, 
           crossvalidate = false;

      po.Register("binary", &binary, "Write output in binary mode");
      po.Register("cross-validate", &crossvalidate, "Perform cross-validation (don't backpropagate)");

      string use_gpu="yes";
      po.Register("use-gpu", &use_gpu, "yes|no|optional, only has effect if compiled with CUDA"); 

      string feature_transform;
      po.Register("feature-transform", &feature_transform, "Feature transform in Nnet format");

      int32 num_stream=8;
      po.Register("num-stream", &num_stream, "---LSTM--- BPTT multi-stream training"); 

      po.Read(argc, argv);

      if (po.NumArgs() != 6-(crossvalidate?1:0)) {
         po.PrintUsage();
         exit(1);
      }

      string feat_rspecifier       = po.GetArg(1),
             label_rspecifier      = po.GetArg(2),
             score_path_rspecifier = po.GetArg(3),
             loss_func_filename    = po.GetArg(4),
             nnet_in_filename      = po.GetArg(5),
             nnet_out_filename;

      if(!crossvalidate){
         nnet_out_filename = po.GetArg(6);
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

      LabelLossBase* loss = LabelLossBase::Read(loss_func_filename);
      if(loss == NULL)
         po.PrintUsage();

      KALDI_LOG << (crossvalidate?"CROSS-VALIDATION":"TRAINING") << " STARTED";

      // ------------------------------------------------------------
      SequentialScorePathReader         score_path_reader(score_path_rspecifier);
      SequentialBaseFloatMatrixReader   feature_reader(feat_rspecifier);
      SequentialUcharVectorReader       label_reader(label_rspecifier);

      MyCuMatrix<BaseFloat> transf_in;

      CuMatrix<BaseFloat> nnet_in;
      MyCuMatrix<BaseFloat> nnet_out;
      MyCuMatrix<BaseFloat> nnet_out_diff;

      vector<int32> flag(1 , 1);
      vector<int32> seqs_length(1, 0);

      Timer time;
      int num_done = 0;

      for(; !feature_reader.Done() && !score_path_reader.Done() && !label_reader.Done();
            ++num_done, feature_reader.Next(), score_path_reader.Next(), label_reader.Next()){

         assert( score_path_reader.Key() == feature_reader.Key() );
         assert( label_reader.Key() == feature_reader.Key() );

         const Matrix<BaseFloat> &feat  = feature_reader.Value();
         const ScorePath::Table  &table = score_path_reader.Value().Value();
         const vector<uchar>     &label = label_reader.Value();

         assert(feat.NumRows() == label.size());

         int seqs_stride = table.size() + 1;

         vector< vector<uchar> > labels_eval;
         labels_eval.reserve(seqs_stride);
         labels_eval.push_back(label);
         for(int i = 0; i < table.size(); ++i){
            assert(table[i].second.size() == label.size());
            labels_eval.push_back(table[i].second);
         }

         vector<uchar> seqs_in(seqs_stride * label.size(), 0);
         for(int i = 0; i < labels_eval.size(); ++i){
            for(int j = 0; j < label.size(); ++j)
               seqs_in[ seqs_stride * j  + i] = labels_eval[i][j] - 1;
         }

         transf_in.Resize(feat.NumRows(), feat.NumCols());
         transf_in.CopyFromMat(feat);
         nnet_transf.Feedforward(transf_in, &nnet_in);

         // setup nnet input
         nnet.SetLabelSeqs(seqs_in, seqs_stride);
         nnet.ResetLstmStreams(flag);

         seqs_length[0] = label.size();
         nnet.SetSeqLengths(seqs_length);

         // propagate nnet output
         nnet.Propagate(nnet_in, &nnet_out);

         assert(nnet_out.NumRows() == label.size() * seqs_stride);
         //nnet_out_diff.Resize(nnet_out.NumRows(), nnet_out.NumCols(), kSetZero);

         loss->Eval(labels_eval, nnet_out, &nnet_out_diff);

         if(!crossvalidate){
            nnet.Backpropagate(nnet_out_diff, NULL);
            if((num_done + 1) % num_stream == 0){
               nnet.Update();
            }
         }

         // report the speed
         if ((num_done+1)%500 == 0) {
            double time_now = time.Elapsed();
            KALDI_VLOG(1) << "After " << num_done+1 << " utterances: time elapsed = "
               << time_now/60 << " min.";
         }
      }

      // last batch
      if(!crossvalidate && num_done % num_stream != 0){
         nnet.Update();
      }

      // after last minibatch : show what happens in network 
      if (kaldi::g_kaldi_verbose_level >= 1) { // vlog-1
         KALDI_VLOG(1) << nnet.InfoPropagate();
         if (!crossvalidate) {
            KALDI_VLOG(1) << nnet.InfoBackPropagate();
            KALDI_VLOG(1) << nnet.InfoGradient();
         }
      }

      if (!crossvalidate) {
         nnet.Write(nnet_out_filename, binary);
      }

      KALDI_LOG << "Done " << num_done << " utterances, "
         << "[" << (crossvalidate?"CROSS-VALIDATION":"TRAINING")
         << ", " << time.Elapsed()/60 << " min.";

      KALDI_LOG << loss->Report();

#if HAVE_CUDA==1
      CuDevice::Instantiate().PrintProfile();
#endif

      return 0;
   } catch(const exception &e) {
      cerr << e.what();
    return -1;
  }
}

