#include "nnet/nnet-trnopts.h"
#include "nnet/nnet-nnet.h"
#include "nnet/nnet-loss.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "base/timer.h"
#include "cudamatrix/cu-device.h"
#include "util.h"
#include "srnnet.h"
#include "kernel.h"
#include <sstream>
#include <pthread.h>



using namespace std;
using namespace kaldi;
using namespace kaldi::nnet1;

int main(int argc, char *argv[]) {

   try {
      string usage;
      usage.append("Structure Recurrent Neural Network calculate scores for all path\n")
         .append("Usage: ").append(argv[0]).append(" [options] <feature-rspecifier> <score-path-rspecifier> <nnet-in> <score-path-wspecifier>\n")
         .append("e.g.: \n")
         .append(" ").append(argv[0]).append(" ark:feat.ark \"ark:lattice-to-nbest --n=1000 ark:test.lat ark:- | lattice-to-vec ark:- ark:- |\" nnet ark:path.ark \n");

      ParseOptions po(usage.c_str());

      string use_gpu="yes";
      po.Register("use-gpu", &use_gpu, "yes|no|optional, only has effect if compiled with CUDA");

      string feature_transform;
      po.Register("feature-transform", &feature_transform, "Feature transform in front of main network (in nnet format)");


      po.Read(argc, argv);

      if (po.NumArgs() != 4) {
         po.PrintUsage();
         exit(1);
      }

      string feat_rspecifier       = po.GetArg(1),
             score_path_rspecifier = po.GetArg(2),
             nnet_in_filename      = po.GetArg(3),
             score_path_wspecifier = po.GetArg(4);


      ScorePathWriter                  score_path_writer(score_path_wspecifier);
      SequentialScorePathReader        score_path_reader(score_path_rspecifier);
      SequentialBaseFloatMatrixReader  feature_reader(feat_rspecifier);

      //Select the GPU
#if HAVE_CUDA==1
      //sleep a while to get lock
      LockSleep(GPU_FILE);
      CuDevice::Instantiate().SelectGpuId(use_gpu);
#endif

      SRNnet nnet;
      nnet.Read(nnet_in_filename);

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

         //nnet.Feedforward(feat, nnet_label_in, &nnet_out);
         nnet.Propagate(feat, nnet_label_in, &nnet_out);

         nnet_out_host.Resize(nnet_out.NumRows(), nnet_out.NumCols(), kUndefined);
         nnet_out_host.CopyFromMat(nnet_out);

         for(int i = 0; i < table.size(); ++i)
            table[i].first = nnet_out_host(i, 0);

         score_path_writer.Write(feature_reader.Key(), table);

         if(num_done % 100 == 0){
            KALDI_LOG << "Done " << num_done;
         }
      }

      // after last minibatch : show what happens in network 
      KALDI_LOG << "###############################";
      KALDI_LOG << nnet.InfoPropagate();

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

