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

      int batch_num=5;
      po.Register("batch-num", &batch_num, "Mini Batch Size");

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


      KALDI_LOG << "GIBBS SAMPLING STARTED";

      Timer time;
      int32 num_done = 0;

      int32 featsN = nnet.InputDim();

      myCuMatrix<BaseFloat> nnet_in(batch_num*max_state, featsN);
      myCuVector<BaseFloat> val(batch_num*max_state);

      vector< myCuMatrix<BaseFloat> >   featArr(batch_num);
      vector< CuIntVector >             pathArr(batch_num);
      vector< string >                  featKey(batch_num);
      vector< BaseFloat >               pathVal(batch_num);

      vector< int32 >               sameCnt(batch_num);

      for ( ; !feature_reader.Done(); ) {

#if HAVE_CUDA==1
         // check the GPU is not overheated
         CuDevice::Instantiate().CheckGpuHealth();
#endif
         // prepare data
         int index;
         for(index = 0; index < batch_num && !feature_reader.Done();
               ++index, feature_reader.Next()){

            const Matrix<BaseFloat> &mat = feature_reader.Value();
            featArr[index].Resize(mat.NumRows(), mat.NumCols(), kUndefined);
            featArr[index].CopyFromMat(mat);
            featKey[index] = feature_reader.Key();
            sameCnt[index] = 0;

            // random start
            vector<int> lab_host(featArr[index].NumRows());
            for(int i = 0; i < featArr[index].NumRows(); ++i)
               lab_host[i] = rand() % max_state;

            pathArr[index].CopyFromVec(lab_host);
         }

         // start sampling 
         for(int i = 0; i < GibbsIter; ++i){
            nnet_in.SetZero();

            for(int j = 0; j < index; ++j){
               makeFeatureCuda(featArr[j], pathArr[j], i % pathArr[j].Dim(), max_state, nnet_in, j);
            }

            nnet.Feedforward(nnet_in, &nnet_out);
            val.CopyColFromMat(nnet_out, 0);

            BaseFloat value;
            for(int j = 0; j < index; ++j){
               if(updateLabelCuda(val, j, pathArr[j], i % pathArr[j].Dim(), max_state, value))
                  sameCnt[j] = 0;
               else
                  sameCnt[j]++;

               pathVal[j] = value;
            }

            bool finished = true;
            for(int j = 0; j < index; ++j)
               if(sameCnt[j] < pathArr[j].Dim() * 2){
                  finished = false;
                  break;
               }
            if(finished) break;
         }

         num_done += index;
         KALDI_LOG << num_done ;

         for(int i = 0; i < index; ++i){
            vector<int> arr;
            pathArr[i].CopyToVec(arr);
            for(int j = 0; j < arr.size(); ++j)
               arr[j] += 1;

            ScorePath::Table table;
            table.push_back(make_pair(pathVal[i], arr));
            score_path_writer.Write(featKey[i], ScorePath(table));
         
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

