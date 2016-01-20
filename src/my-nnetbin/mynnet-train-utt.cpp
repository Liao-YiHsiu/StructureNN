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
#include "nnet-my-loss.h"
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

      CuMatrix<BaseFloat> nnet_in;
      CuMatrix<BaseFloat> nnet_out;
      CuMatrix<BaseFloat> nnet_out_diff;
      vector< CuMatrix<BaseFloat> > nnet_out_diff_arr(num_stream);
      vector< CuMatrix<BaseFloat> > features(num_stream);

      Timer time;
      int seqs_stride = -1;
      int num_done = 0;


      while(1){

         vector< CuMatrix<BaseFloat> >     features(num_stream);
         vector< vector< vector<uchar> > > label_arr(num_stream);
         int streams = 0;
         int max_length = 0;

         for(; streams < num_stream &&
               !feature_reader.Done() && !score_path_reader.Done() && !label_reader.Done();
               ++streams, feature_reader.Next(), score_path_reader.Next(), label_reader.Next()){

            assert( score_path_reader.Key() == feature_reader.Key() );
            assert( label_reader.Key() == feature_reader.Key() );

            const Matrix<BaseFloat> &feat  = feature_reader.Value();
            const ScorePath::Table  &table = score_path_reader.Value().Value();
            const vector<uchar>     &label = label_reader.Value();

            if(max_length < label.size()) max_length = label.size();

            nnet_transf.Feedforward(CuMatrix<BaseFloat>(feat), &features[streams]);

            vector< vector<uchar> > & seqs = label_arr[streams];

            seqs.reserve(1 + table.size());

            seqs.push_back(label);

            for(int i = 0; i < table.size(); ++i)
               seqs.push_back(table[i].second);

            if(seqs_stride < 0){
               seqs_stride = seqs.size();
            }
            assert(seqs_stride == seqs.size());
            assert(feat.NumRows() == label.size());

         }

         if(streams == 0) break;

         nnet_in.Resize(max_length * streams, features[0].NumCols(), kSetZero);
         fillin(nnet_in, features, streams);

         vector<int32> label_in(max_length * streams * seqs_stride, 0);
         vector<int32> seq_length(streams, 0);

         // re-arrange label index
#pragma omp for
         for(int i = 0; i < streams; ++i){
            const vector< vector<uchar> > &label = label_arr[i];
            for(int j = 0; j < label.size(); ++j)
               for(int k = 0; k < label[j].size(); ++k)
                  label_in[ seqs_stride * streams * k  + seqs_stride * i + j] =
                     label[j][k] - 1;

            seq_length[i] = label[0].size();
         }

         // setup nnet input
         nnet.SetLabelSeqs(label_in, seqs_stride);

         vector<int32> flag(streams , 1);
         nnet.ResetLstmStreams(flag);
         nnet.SetSeqLengths(seq_length);

         // propagate nnet output
         nnet.Propagate(nnet_in, &nnet_out);

         assert(nnet_out.NumRows() == max_length * seqs_stride * streams);
         nnet_out_diff.Resize(nnet_out.NumRows(), nnet_out.NumCols(), kSetZero);

         for(int i = 0; i < streams; ++i){
            loss->Eval(label_arr[i],
                  nnet_out.RowRange(i*max_length*seqs_stride, max_length*seqs_stride), &nnet_out_diff_arr[i]);
            nnet_out_diff.RowRange(i*max_length*seqs_stride, max_length*seqs_stride).CopyFromMat(nnet_out_diff_arr[i]);
         }

         if(!crossvalidate){
            nnet.Backpropagate(nnet_out_diff, NULL);
         }

         num_done += streams;

         // report the speed
         if ((num_done+1)%10 == 0) {
            double time_now = time.Elapsed();
            KALDI_VLOG(1) << "After " << num_done << " utterances: time elapsed = "
               << time_now/60 << " min.";
         }
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

