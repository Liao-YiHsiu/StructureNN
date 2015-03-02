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
    srand(seed);

    int mini_batch=256;
    po.Register("mini-batch", &mini_batch, "Mini Batch Size");

    int max_state = 48;
    po.Register("max-state", &max_state, "max state ID");

    int GibbsIter = 10000;
    po.Register("GibbsIter", &GibbsIter, "Gibbs Sampling Iteration");

    po.Read(argc, argv);

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

    Nnet nnet;
    nnet.Read(model_filename);

    nnet.SetDropoutRetention(1.0);

    Matrix<BaseFloat> nnet_out_host;
    Posterior targets;
    CuMatrix<BaseFloat> nnet_out, nnet_in;


    KALDI_LOG << "GIBBS SAMPLING STARTED";

    Timer time;
    double time_now = 0;


    kaldi::int64 tot_t = 0;
    int32 num_done = 0;
    int32 featsN = nnet.InputDim();

    int batch_num = mini_batch / max_state;

    vector<BaseFloat> probArr(max_state);

    vector< Matrix<BaseFloat> >   featArr(batch_num);
    vector< string >              featKey(batch_num);
    vector< BaseFloat >           pathVal(batch_num);
    vector<vector<int32> >        pathArr(batch_num);

    vector< int32 >               sameCnt(batch_num);
    vector< int32 >               old(batch_num);

    for ( ; !feature_reader.Done();) {

#if HAVE_CUDA==1
       // check the GPU is not overheated
       CuDevice::Instantiate().CheckGpuHealth();
#endif

       // prepare feature and start path
       int index;
       for(index = 0; index < batch_num && !feature_reader.Done();
             ++index, feature_reader.Next()){


          featArr[index] = feature_reader.Value();
          featKey[index] = feature_reader.Key();
          sameCnt[index]  = 0;

          // random start
          pathArr[index].resize(featArr[index].NumRows());
          for(int j = 0; j < pathArr[index].size(); ++j)
             pathArr[index][j] = rand() % max_state + 1;

          assert((featArr[index].NumCols() + max_state)*max_state == featsN);
       }

       // start training
       for(int i = 0; i < GibbsIter; ++i){
          for(int j = 0; j < index; ++j)
             old[j] = pathArr[j][i % pathArr[j].size()];

          //vector<pthread_t> threads(index*max_state);
          //vector<FData>     fData(index*max_state);
          vector<pthread_t> threads(index);
          vector<FData>     fData(index);
          int rc;

          Matrix<BaseFloat> feats(batch_num * max_state, featsN);
          for(int j = 0; j < index; ++j){
            // makeFeatureBatch(featArr[j], pathArr[j], i % pathArr[j].size(), max_state,
            //       feats.RowRange(j*max_state, max_state));

             fData[j].feat     = &featArr[j];
             fData[j].path     = &pathArr[j];
             fData[j].maxState = max_state;
             fData[j].mat      = new SubMatrix<BaseFloat>(feats.RowRange(j*max_state, max_state));
             fData[j].chgID    = i % pathArr[j].size();

             // use threads
             rc = pthread_create(&threads[j], NULL, makeFeatureP, (void *) &fData[j]);
             assert(0 == rc);
             //rc = pthread_create(&threads[j*max_state+k], NULL, makeFeatureP, (void *) &fData[ID]);

             //for(int k = 0; k < max_state; ++k){
             //   //pathArr[j][ i % pathArr[j].size() ] = k + 1;
             //   //makeFeature(featArr[j], pathArr[j], max_state, feats.Row(j*max_state + k));

             //}
          }

          for(int j = 0; j < threads.size(); ++j){
             // block until thread i completes
             rc = pthread_join(threads[j], NULL);
             assert(0 == rc);

             delete fData[j].mat;
          }

          nnet_in.Resize(feats.NumRows(), feats.NumCols());
          nnet_in.CopyFromMat(feats);

          nnet.Feedforward(nnet_in, &nnet_out);

          //download from GPU
          nnet_out_host.Resize(nnet_out.NumRows(), nnet_out.NumCols());
          nnet_out.CopyToMat(&nnet_out_host);

          assert(nnet_out_host.NumRows() == feats.NumRows());

          for(int j = 0; j < index; ++j){
             for(int k = 0; k < max_state; ++k){
                probArr[k] = nnet_out_host(j*max_state + k, 0); 
             }
             int32 choose = best(probArr);
             pathArr[j][ i % pathArr[j].size() ] = choose + 1;
             pathVal[j] = probArr[choose];

             if(choose + 1 == old[j])
                sameCnt[j]++;
             else
                sameCnt[j] = 0;
          }
          tot_t += feats.NumRows();

          if(i % 1000 == 0)
             KALDI_LOG << num_done << "\t" << i ; 

          bool stop = true;
          for(int j = 0; j < index; ++j)
             if(sameCnt[j] < pathArr[j].size()){
                stop = false;
                break;
             }
                
          // early stop
          if(stop) break;
       }

       for(int j = 0; j < index; ++j){
          ScorePath::Table table;
          table.push_back(make_pair(pathVal[j], pathArr[j]));
          score_path_writer.Write(featKey[j], ScorePath(table));
       }

       // progress log
       if (num_done % 100 == 0) {
          time_now = time.Elapsed();
          KALDI_VLOG(1) << "After " << num_done << " utterances: time elapsed = "
             << time_now/60 << " min; processed " << tot_t/time_now
             << " frames per second.";
       }
       num_done+=index;

       if(feature_reader.Done())
          break;

    }

    // final message
    KALDI_LOG << "Done " << num_done << " files" 
              << " in " << time.Elapsed()/60 << "min," 
              << " (fps " << tot_t/time.Elapsed() << ")"; 

#if HAVE_CUDA==1
    CuDevice::Instantiate().PrintProfile();
#endif

    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}



