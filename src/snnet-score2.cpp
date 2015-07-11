#include "nnet/nnet-trnopts.h"
#include "nnet/nnet-nnet.h"
#include "nnet/nnet-loss.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "base/timer.h"
#include "cudamatrix/cu-device.h"
#include "svm.h"
#include "snnet.h"
#include "kernel.h"
#include <sstream>
#include <pthread.h>



using namespace std;
using namespace kaldi;
using namespace kaldi::nnet1;

int main(int argc, char *argv[]) {

   try {
      string usage;
      usage.append("Structure Neural Network calculate scores for all path\n")
         .append("Usage: ").append(argv[0]).append(" [options] <feature-rspecifier> <score-path-rspecifier> <nnet1-in> <nnet2-in> <stateMax> <score-path-wspecifier>\n")
         .append("e.g.: \n")
         .append(" ").append(argv[0]).append(" ark:feat.ark \"ark:lattice-to-nbest --n=1000 ark:test.lat ark:- | lattice-to-vec ark:- ark:- |\" nnet1 nnet2 48 ark:path.ark \n");

      ParseOptions po(usage.c_str());

      string use_gpu="yes";
      po.Register("use-gpu", &use_gpu, "yes|no|optional, only has effect if compiled with CUDA");

      string feature_transform;
      po.Register("feature-transform", &feature_transform, "Feature transform in front of main network (in nnet format)");


      po.Read(argc, argv);

      if (po.NumArgs() != 6) {
         po.PrintUsage();
         exit(1);
      }

      string feat_rspecifier       = po.GetArg(1),
             score_path_rspecifier = po.GetArg(2),
             nnet1_in_filename     = po.GetArg(3),
             nnet2_in_filename     = po.GetArg(4);
      int    stateMax              = atoi(po.GetArg(5).c_str());
      string score_path_wspecifier = po.GetArg(6);


      ScorePathWriter                  score_path_writer(score_path_wspecifier);
      SequentialScorePathReader        score_path_reader(score_path_rspecifier);
      SequentialBaseFloatMatrixReader  feature_reader(feat_rspecifier);

      //Select the GPU
#if HAVE_CUDA==1
      CuDevice::Instantiate().SelectGpuId(use_gpu);
#endif

      SNnet nnet;
      nnet.Read(nnet1_in_filename, nnet2_in_filename, stateMax);

      nnet.SetDropoutRetention(1.0);

      if (feature_transform != "") {
         Nnet nnet_transf;
         nnet_transf.Read(feature_transform);
         nnet.SetTransform(nnet_transf);
      }

      KALDI_LOG << "SNNet pick best started.";

      Timer time;
      int32 num_done = 0;

      CuMatrix<BaseFloat> nnet_out;
      Matrix<BaseFloat> nnet_out_host;

      for ( ; !feature_reader.Done() && !score_path_reader.Done(); 
            feature_reader.Next(), score_path_reader.Next(), num_done++) {

         assert(feature_reader.Key() == score_path_reader.Key());

#if HAVE_CUDA==1
         // check the GPU is not overheated
         CuDevice::Instantiate().CheckGpuHealth();
#endif
         CuMatrix<BaseFloat>      feat(feature_reader.Value());
         ScorePath::Table table = score_path_reader.Value().Value();

         vector<vector<uchar> * >      nnet_label_in;

         for(int i = 0; i < table.size(); ++i){
            nnet_label_in.push_back(&table[i].second);
         }

         nnet.Feedforward(feat, nnet_label_in, &nnet_out);

         nnet_out_host.Resize(nnet_out.NumRows(), nnet_out.NumCols(), kUndefined);
         nnet_out_host.CopyFromMat(nnet_out);

         for(int i = 0; i < table.size(); ++i)
            table[i].first = nnet_out_host(i, 0);

         score_path_writer.Write(feature_reader.Key(), table);

         if(num_done % 10 == 0){
            KALDI_LOG << "Done " << num_done;
         }
      }

      // final message
      KALDI_LOG << "Done " << num_done << " files" 
         << " in " << time.Elapsed()/60 << "min";

#if HAVE_CUDA==1
      CuDevice::Instantiate().PrintProfile();
#endif

      return 0;
   } catch(const std::exception &e) {
      std::cerr << e.what();
      return -1;
   }
}

