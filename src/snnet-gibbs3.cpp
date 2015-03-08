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
      usage.append("Structure Neural Network predict with Gibbs Sampling.\n")
         .append("Use Gibbs Sampling to find best path.\n")
         .append("Usage: ").append(argv[0]).append(" [options] <feature-rspecifier> <model-in> <score-path-wspecifier>\n")
         .append("e.g.: \n")
         .append(" ").append(argv[0]).append(" ark:feat.ark nnet ark:path.ark \n");

      ParseOptions po(usage.c_str());

      string use_gpu="yes";
      //po.Register("use-gpu", &use_gpu, "yes|no|optional, only has effect if compiled with CUDA");

      int seed=777;
      po.Register("seed", &seed, "Random Seed Number.");

      int max_state = 48;
      po.Register("max-state", &max_state, "max state ID");

      int GibbsIter = 10000;
      po.Register("GibbsIter", &GibbsIter, "Gibbs Sampling Iteration");

      po.Read(argc, argv);
      srand(seed);

      if (po.NumArgs() != 3) {
         po.PrintUsage();
         exit(1);
      }

      string feat_rspecifier       = po.GetArg(1),
             model_filename        = po.GetArg(2),
             score_path_wspecifier = po.GetArg(3);


      ScorePathWriter                  score_path_writer(score_path_wspecifier);
      SequentialBaseFloatMatrixReader  feature_reader(feat_rspecifier);

      //Select the GPU
#if HAVE_CUDA==1
      CuDevice::Instantiate().SelectGpuId(use_gpu);
      CuDevice::Instantiate().DisableCaching();
#endif

      Nnet nnet;
      nnet.Read(model_filename);

      nnet.SetDropoutRetention(1.0);

      Posterior targets;
      CuMatrix<BaseFloat> nnet_out;
      myCuMatrix<BaseFloat> nnet_in;


      KALDI_LOG << "GIBBS SAMPLING STARTED";

      Timer time;
      int32 num_done = 0;

      for ( ; !feature_reader.Done(); feature_reader.Next()) {

#if HAVE_CUDA==1
         // check the GPU is not overheated
         CuDevice::Instantiate().CheckGpuHealth();
#endif

         // one at a time.
         //const Matrix<BaseFloat>& mat = feature_reader.Value();
         CuMatrix<BaseFloat> feat(feature_reader.Value());

         vector<int> lab_host(feat.NumRows());
         for(int i = 0; i < feat.NumRows(); ++i)
            lab_host[i] = rand() % max_state;

         CuIntVector          lab(lab_host);
         CuVector<BaseFloat>  val(lab_host.size() * max_state);

         BaseFloat value;

         myCuMatrix<BaseFloat> feat_my(feat);

         int i;
         for(i = 0; i < GibbsIter; ++i){
            makeFeatureCuda(feat_my, lab, max_state, nnet_in);
            nnet.Feedforward(nnet_in, &nnet_out);
            val.CopyColFromMat(nnet_out, 0);
            if(!updateLabelCuda(val, lab, max_state, value))
               break;

         }
         num_done++;
         KALDI_LOG << num_done << "\t" << i;

         vector<int> arr;
         lab.CopyToVec(arr);
         for(i = 0; i < arr.size(); ++i)
            arr[i] += 1;

         ScorePath::Table table;
         table.push_back(make_pair(value, arr));
         score_path_writer.Write(feature_reader.Key(), ScorePath(table));
         
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

