#include "nnet/nnet-trnopts.h"
#include "nnet/nnet-nnet.h"
#include "nnet/nnet-loss.h"
#include "nnet/nnet-randomizer.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "base/timer.h"
#include "cudamatrix/cu-device.h"
#include "svm.h"
#include <sstream>
#include <pthread.h>


using namespace std;
using namespace kaldi;
using namespace kaldi::nnet1;

typedef struct{
   Matrix<BaseFloat>  *mat;
   BaseFloat          *val;
   vector<int32>      *seq;
} GTask;

pthread_mutex_t task_mutex;
pthread_mutex_t cuda_mutex;
pthread_mutex_t print_mutex;
int max_state;
int32 featsN;
int GibbsIter;
int32 num_done = 0;
Nnet nnet;

void* runner(void *param){
   vector<GTask> *taskArr = (vector<GTask>*)param;

   Matrix<BaseFloat> feats(max_state, featsN);
   Matrix<BaseFloat> nnet_out_host;
   CuMatrix<BaseFloat> nnet_out, nnet_in;
   vector<BaseFloat> probArr(max_state);

   while(true){
      GTask task;

      pthread_mutex_lock(&task_mutex);
      {
         if(taskArr->size() == 0){
            pthread_mutex_unlock(&task_mutex);
            return NULL;
         }
         task = taskArr->back();
         taskArr->pop_back();
      }
      pthread_mutex_unlock(&task_mutex);

      vector<int32>     &path = *(task.seq);
      Matrix<BaseFloat> &feat = *(task.mat);
      BaseFloat         &val  = *(task.val);

      int32 old;
      int32 sameCnt = 0;

      // start training
      for(int i = 0; i < GibbsIter; ++i){
         for(int j = 0; j < path.size(); ++j){
            old = path[j];

            // make feature
            for(int k = 0; k < max_state; ++k){
               path[j] = k + 1;
               makeFeature(feat, path, max_state, feats.Row(k));
            }


            pthread_mutex_lock(&cuda_mutex);
            {

#if HAVE_CUDA==1
               // check the GPU is not overheated
               CuDevice::Instantiate().CheckGpuHealth();
#endif
               nnet_in.Resize(feats.NumRows(), feats.NumCols());
               nnet_in.CopyFromMat(feats);

               nnet.Feedforward(nnet_in, &nnet_out);

               //download from GPU
               nnet_out_host.Resize(nnet_out.NumRows(), nnet_out.NumCols());
               nnet_out.CopyToMat(&nnet_out_host);
            }
            pthread_mutex_unlock(&cuda_mutex);

            assert(nnet_out_host.NumRows() == feats.NumRows());

            // choose max
            for(int k = 0; k < max_state; ++k){
               probArr[k] = nnet_out_host(k, 0); 
            }
            int32 choose = best(probArr);
            path[j] = choose + 1;
            val = probArr[choose];

            if(choose + 1 == old)
               sameCnt++;
            else
               sameCnt = 0;


         }

         if(sameCnt >= path.size()){
            break;
         }

      }

      pthread_mutex_lock(&print_mutex);
      {
         num_done++;
         KALDI_LOG << num_done; 
      }
      pthread_mutex_unlock(&print_mutex);

   }
}

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
    po.Register("use-gpu", &use_gpu, "yes|no|optional, only has effect if compiled with CUDA");

    int seed=777;
    po.Register("seed", &seed, "Random Seed Number.");

    int mini_batch=256;
    po.Register("mini-batch", &mini_batch, "Mini Batch Size");

    max_state = 48;
    po.Register("max-state", &max_state, "max state ID");

    GibbsIter = 50;
    po.Register("GibbsIter", &GibbsIter, "Gibbs Sampling Iteration");

    int thread_num = 10;
    po.Register("thread-num", &thread_num, "number of threads to speed up.");

    po.Read(argc, argv);
    srand(seed);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    string feat_rspecifier  = po.GetArg(1),
      model_filename        = po.GetArg(2),
      score_path_wspecifier       = po.GetArg(3);


    ScorePathWriter                  score_path_writer(score_path_wspecifier);
    SequentialBaseFloatMatrixReader  feature_reader(feat_rspecifier);

    //Select the GPU
#if HAVE_CUDA==1
    CuDevice::Instantiate().SelectGpuId(use_gpu);
    CuDevice::Instantiate().DisableCaching();
#endif

    nnet.Read(model_filename);
    nnet.SetDropoutRetention(1.0);

    KALDI_LOG << "GIBBS SAMPLING STARTED";

    Timer time;
    num_done = 0;
    featsN = nnet.InputDim();


    vector< Matrix<BaseFloat> >   featArr;
    vector< string >              featKey;
    vector< BaseFloat >           pathVal; // score
    vector< vector<int32> >       pathArr;

    vector< GTask >               taskArr;

    // put tasks into thread pull;
    for(; !feature_reader.Done(); feature_reader.Next()){
       const Matrix<BaseFloat>& mat = feature_reader.Value();
       featArr.push_back(mat);
       featKey.push_back(feature_reader.Key());
       pathVal.push_back(0);

       vector<int32> path(mat.NumRows());
       for(int j = 0; j < path.size(); ++j)
          path[j] = rand() % max_state + 1;

       pathArr.push_back(path);

       GTask task;
       task.mat = &featArr.back(); 
       task.val = &pathVal.back();
       task.seq = &pathArr.back();

       taskArr.push_back(task);
    }

    pthread_mutex_init(&task_mutex, NULL);
    pthread_mutex_init(&cuda_mutex, NULL);
    pthread_mutex_init(&print_mutex, NULL);

    vector<pthread_t> threads(thread_num);
    // new threads to do tasks.
    for(int i = 0; i < threads.size(); ++i){
       int rc = pthread_create(&threads[i], NULL, runner, (void *) &taskArr);
       assert(0 == rc);
    }

    // wait until threads finished.
    for(int j = 0; j < threads.size(); ++j){
       int rc = pthread_join(threads[j], NULL);
       assert(0 == rc);
    }
    pthread_mutex_destroy(&task_mutex);
    pthread_mutex_destroy(&cuda_mutex);
    pthread_mutex_destroy(&print_mutex);

    for(int j = 0; j < pathArr.size(); ++j){
       ScorePath::Table table;
       table.push_back(make_pair(pathVal[j], pathArr[j]));
       score_path_writer.Write(featKey[j], ScorePath(table));
    }


    // final message
    KALDI_LOG << "Done " << num_done << " files" 
              << " in " << time.Elapsed()/60 << "min,";

#if HAVE_CUDA==1
    CuDevice::Instantiate().PrintProfile();
#endif

    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}



