#include "nnet/nnet-trnopts.h"
#include "nnet/nnet-nnet.h"
#include "nnet/nnet-loss.h"
#include "nnet/nnet-randomizer.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "base/timer.h"
#include "cudamatrix/cu-device.h"
#include "svm.h"
#include "kernel.h"
#include <sstream>
#include <pthread.h>



using namespace std;
using namespace kaldi;
using namespace kaldi::nnet1;




int main(int argc, char *argv[]) {

   try {
      string usage;
      usage.append("Structure Neural Network choose the best score from all path\n")
         .append("Usage: ").append(argv[0]).append(" [options] <feature-rspecifier> <score-path-rspecifier> <model-in> <score-path-wspecifier>\n")
         .append("e.g.: \n")
         .append(" ").append(argv[0]).append(" ark:feat.ark \"ark:lattice-to-nbest --n=1000 ark:test.lat ark:- | lattice-to-vec ark:- ark:- |\" nnet ark:path.ark \n");

      ParseOptions po(usage.c_str());

      string use_gpu="yes";
      po.Register("use-gpu", &use_gpu, "yes|no|optional, only has effect if compiled with CUDA");

      int seed=777;
      po.Register("seed", &seed, "Random Seed Number.");

      int max_state = 48;
      po.Register("max-state", &max_state, "max state ID");


      po.Read(argc, argv);
      srand(seed);

      if (po.NumArgs() != 4) {
         po.PrintUsage();
         exit(1);
      }

      string feat_rspecifier       = po.GetArg(1),
             score_path_rspecifier = po.GetArg(2),
             model_filename        = po.GetArg(3),
             score_path_wspecifier = po.GetArg(4);


      ScorePathWriter                  score_path_writer(score_path_wspecifier);
      SequentialScorePathReader        score_path_reader(score_path_rspecifier);
      SequentialBaseFloatMatrixReader  feature_reader(feat_rspecifier);

      //Select the GPU
#if HAVE_CUDA==1
      CuDevice::Instantiate().SelectGpuId(use_gpu);
#endif

      Nnet nnet;
      nnet.Read(model_filename);

      nnet.SetDropoutRetention(1.0);

      Posterior targets;


      KALDI_LOG << "SNNet pick best started.";

      Timer time;
      int32 num_done = 0;

      int32 featsN = nnet.InputDim();

      CuMatrix<BaseFloat> nnet_in;
      CuMatrix<BaseFloat> nnet_out;
      Matrix<BaseFloat> nnet_in_host;
      Matrix<BaseFloat> nnet_out_host;
      Vector<BaseFloat> val;

      for ( ; !feature_reader.Done() && !score_path_reader.Done(); 
            feature_reader.Next(), score_path_reader.Next(), num_done++) {

         assert(feature_reader.Key() == score_path_reader.Key());

#if HAVE_CUDA==1
         // check the GPU is not overheated
         CuDevice::Instantiate().CheckGpuHealth();
#endif
         const Matrix<BaseFloat> &mat  = feature_reader.Value();
         const ScorePath::Table &table = score_path_reader.Value().Value();

         nnet_in.Resize(table.size(), featsN, kUndefined);
         nnet_in_host.Resize(table.size(), featsN, kSetZero);
         val.Resize(table.size(), kUndefined);


         for(int i = 0; i < table.size(); ++i){
            makeFeature(mat, table[i].second, max_state, nnet_in_host.Row(i));
         }

         nnet_in.CopyFromMat(nnet_in_host);
         nnet.Feedforward(nnet_in, &nnet_out);
         nnet_out_host.Resize(nnet_out.NumRows(), nnet_out.NumCols(), kUndefined);
         nnet_out_host.CopyFromMat(nnet_out);
         val.CopyColFromMat(nnet_out_host, 0);

         int maxIdx = rand()%val.Dim();
         float max  = val(maxIdx);

         for(int i = 0; i < val.Dim(); ++i)
            if( val(i) > max){
               maxIdx = i;
               max    = val(i);
            }

         ScorePath tmpscore;
         tmpscore.Value().push_back(make_pair(max, table[maxIdx].second));
         score_path_writer.Write(feature_reader.Key(), tmpscore);
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

