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
   string             *key;
   bool               done;
} GTask;

pthread_mutex_t   task_mutex;
pthread_mutex_t   outer_mutex;
pthread_mutex_t   print_mutex;

pthread_mutex_t   cuda_mutex;
pthread_cond_t    cuda_cond;
pthread_mutex_t   cuda_finish_mutex;
pthread_cond_t    cuda_finish_cond;

int               max_state;
int               GibbsIter;
int32             featsN;
int               running;
int32             num_done;
Matrix<BaseFloat> *pnnet_in_host, *pnnet_out_host; 
bool              waiting;
bool              finished;

void* runner(void *param);

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
           model_filename   = po.GetArg(2),
      score_path_wspecifier = po.GetArg(3);


    ScorePathWriter                  score_path_writer(score_path_wspecifier);
    SequentialBaseFloatMatrixReader  feature_reader(feat_rspecifier);

    //Select the GPU
#if HAVE_CUDA==1
    CuDevice::Instantiate().SelectGpuId(use_gpu);
#endif

    Nnet nnet;
    nnet.Read(model_filename);
    nnet.SetDropoutRetention(1.0);

    featsN = nnet.InputDim();


    KALDI_LOG << "GIBBS SAMPLING STARTED";

    Timer time;
    num_done = 0;
    //featsN = nnet.InputDim();


    //vector< Matrix<BaseFloat> >   featArr;
    vector< string >              featKey;
    //vector< BaseFloat >           pathVal; // score
    //vector< vector<int32> >       pathArr;

    vector< GTask >               taskArr;

    // put tasks into thread pull;
    for(; !feature_reader.Done(); feature_reader.Next()){
       const Matrix<BaseFloat>& mat = feature_reader.Value();
       //featArr.push_back(mat);
       featKey.push_back(feature_reader.Key());
       //pathVal.push_back(0);

       vector<int32> path(mat.NumRows());
       for(int j = 0; j < path.size(); ++j)
          path[j] = rand() % max_state + 1;
          //path[j] = 1;

       //pathArr.push_back(path);

       GTask task;
       task.mat  = new Matrix<BaseFloat>(mat); 
       task.val  = new BaseFloat(0);
       task.seq  = new vector<int32>(path);
       task.key  = new string(feature_reader.Key());
       task.done = false;
       //task.nnet = &nnet;

       taskArr.push_back(task);
    }
    
    KALDI_LOG << "READ ALL SAMPLE";

    pthread_mutex_init(&task_mutex, NULL);
    pthread_mutex_init(&outer_mutex, NULL);
    pthread_mutex_init(&cuda_mutex, NULL);
    pthread_cond_init (&cuda_cond, NULL);
    pthread_mutex_init(&cuda_finish_mutex, NULL);
    pthread_cond_init (&cuda_finish_cond, NULL);
    pthread_mutex_init(&print_mutex, NULL);

    vector<pthread_t> threads(thread_num);

    waiting  = false;
    finished = false;
    running  = threads.size();
    // new threads to do tasks.
    //runner(&taskArr);
    //CU_SAFE_CALL(cublasGetError());
    for(int i = 0; i < threads.size(); ++i){
       int rc = pthread_create(&threads[i], NULL, runner, (void *) &taskArr);
       assert(0 == rc);
    }
    CuMatrix<BaseFloat> nnet_in, nnet_out;

    // handling NN forward
    pthread_mutex_lock(&cuda_mutex);
    while(true){
       while(!waiting){
          pthread_cond_wait(&cuda_cond, &cuda_mutex);
          if(running == 0) break;
       }
       if(running == 0){
          break;
       }
       waiting = false;

//#if HAVE_CUDA==1
       // check the GPU is not overheated
//       CuDevice::Instantiate().CheckGpuHealth();
//#endif

       nnet_in.Resize(pnnet_in_host->NumRows(), pnnet_in_host->NumCols());

       nnet_in.CopyFromMat(*pnnet_in_host);

       nnet.Feedforward(nnet_in, &nnet_out);

       pnnet_out_host->Resize(nnet_out.NumRows(), nnet_out.NumCols());
       pnnet_out_host->CopyFromMat(nnet_out);

       pthread_mutex_lock(&cuda_finish_mutex);
       {
          finished = true;
          pthread_cond_signal(&cuda_finish_cond);
       }
       pthread_mutex_unlock(&cuda_finish_mutex);

    }
    pthread_mutex_unlock(&cuda_mutex);

    // wait until threads finished.
    for(int j = 0; j < threads.size(); ++j){
       int rc = pthread_join(threads[j], NULL);
       assert(0 == rc);
    }
    pthread_mutex_destroy(&task_mutex);
    pthread_mutex_destroy(&outer_mutex);
    pthread_mutex_destroy(&cuda_mutex);
    pthread_cond_destroy (&cuda_cond);
    pthread_mutex_destroy(&cuda_finish_mutex);
    pthread_cond_destroy (&cuda_finish_cond);
    pthread_mutex_destroy(&print_mutex);

    //for(int j = 0; j < pathArr.size(); ++j){
    for(int i = 0; i < taskArr.size(); ++i){
       ScorePath::Table table;
       table.push_back(make_pair(*taskArr[i].val, *taskArr[i].seq));
       score_path_writer.Write(featKey[i], ScorePath(table));

       delete taskArr[i].mat;
       delete taskArr[i].val;
       delete taskArr[i].seq;
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

void* runner(void *param){


   vector<GTask> *taskArr = (vector<GTask>*)param;

   Matrix<BaseFloat>* feats = new Matrix<BaseFloat>(max_state, featsN);
   Matrix<BaseFloat>* nnet_out_host = new Matrix<BaseFloat>();
   //CuMatrix<BaseFloat> nnet_out, nnet_in;
   vector<BaseFloat> probArr(max_state);

   while(true){
      GTask task;

      pthread_mutex_lock(&task_mutex);
      {
         int index = -1;
         for(int i = 0; i < taskArr->size(); ++i)
            if(!(*taskArr)[i].done){
               index = i;
               break;
            }

         if(index == -1){
            running--;

            if(running == 0){
               pthread_mutex_lock(&cuda_mutex);
               pthread_cond_signal(&cuda_cond);
               pthread_mutex_unlock(&cuda_mutex);
            }

            pthread_mutex_unlock(&task_mutex);
            delete feats;
            delete nnet_out_host;
            return NULL;
         }
         task = (*taskArr)[index];
         (*taskArr)[index].done = true;
      }
      pthread_mutex_unlock(&task_mutex);

      vector<int32>     &path = *(task.seq);
      Matrix<BaseFloat> &feat = *(task.mat);
      BaseFloat         &val  = *(task.val);
      //Nnet              &nnet = *(task.nnet);

      int32 old;
      int32 sameCnt = 0;

      // start training
      for(int i = 0; i < GibbsIter; ++i){
         for(int j = 0; j < path.size(); ++j){
            old = path[j];

            // make feature
            for(int k = 0; k < max_state; ++k){
               path[j] = k + 1;
               makeFeature(feat, path, max_state, feats->Row(k));
            }

            //nnet_in.Resize(feats.NumRows(), feats.NumCols());
            //nnet_in.CopyFromMat(feats);

            pthread_mutex_lock(&outer_mutex);
            {

               pthread_mutex_lock(&cuda_mutex);
               {

                  pnnet_in_host  = feats;
                  pnnet_out_host = nnet_out_host;

                  waiting = true;
                  pthread_cond_signal(&cuda_cond);

               }
               pthread_mutex_unlock(&cuda_mutex);

               // wait until finished
               pthread_mutex_lock(&cuda_finish_mutex);
               while(!finished){
                  pthread_cond_wait(&cuda_finish_cond, &cuda_finish_mutex);
               }
               finished = false;
               pthread_mutex_unlock(&cuda_finish_mutex);

            }
            pthread_mutex_unlock(&outer_mutex);

            assert(nnet_out_host->NumRows() == feats->NumRows());

            // choose max
            for(int k = 0; k < max_state; ++k){
               probArr[k] = (*nnet_out_host)(k, 0); 
            }
            int32 choose = best(probArr);
            path[j] = choose + 1;
            assert(val <= probArr[choose] + 0.001);
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
         //if(num_done % 100 == 0)
            KALDI_LOG << num_done; 
      }
      pthread_mutex_unlock(&print_mutex);

   }

   assert(false);
}
