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

using namespace std;
using namespace kaldi;
using namespace kaldi::nnet1;

int main(int argc, char *argv[]) {
  
  try {
    string usage;
    usage.append("Structure Neural Network predict with Gibbs Sampling.\n")
       .append("Random start path, use Gibbs Sampling to find best path.\n")
       .append("Usage: ").append(argv[0]).append(" [options] <feature-rspecifier> <model-in> <path-wspecifier>\n")
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
      path_wspecifier       = po.GetArg(3);


    Int32VectorWriter                path_writer(path_wspecifier);
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

    Matrix<BaseFloat> feats(batch_num * max_state, featsN);
    //KALDI_LOG << "matrix" << feats.Sum();

    vector<BaseFloat> probArr(max_state);

    vector< Matrix<BaseFloat> > featArr(batch_num);
    vector< string > featKey(batch_num);

    for ( ; !feature_reader.Done(); feature_reader.Next()) {

#if HAVE_CUDA==1
          // check the GPU is not overheated
          CuDevice::Instantiate().CheckGpuHealth();
#endif

       //const Matrix<BaseFloat> &feat  = feature_reader.Value();
       int index;
       for(index = 0; index < batch_num && !feature_reader.Done();
             ++index, feature_reader.Next()){
          //const Matrix<BaseFloat> &mat = feature_reader.Value();
          //featArr[index].Resize(mat.NumRows(), mat.NumCols());
          //featArr[index].CopyFromMat(mat);
          featArr[index] = feature_reader.Value();
          featKey[index] = feature_reader.Key();

          assert((featArr[index].NumCols() + max_state)*max_state == featsN);
       }

       vector<vector<int32> > pathArr(index);
       for(int i = 0; i < index; ++i){
          pathArr[i].resize(featArr[i].NumRows());
          for(int j = 0; j < pathArr[i].size(); ++j)
             pathArr[i][j] = rand() % max_state + 1;
       }

       for(int i = 0; i < GibbsIter; ++i){

          for(int j = 0; j < index; ++j){
             for(int k = 0; k < max_state; ++k){
                pathArr[j][ i % pathArr[j].size() ] = k + 1;
                makeFeature(featArr[j], pathArr[j], max_state, feats.Row(j*max_state + k));
             }
          }
          nnet_in.Resize(feats.NumRows(), feats.NumCols());
          nnet_in.CopyFromMat(feats);

          //nnet.Feedforward(CuMatrix<BaseFloat>(feats), &nnet_out);
          nnet.Feedforward(nnet_in, &nnet_out);

          //download from GPU
          nnet_out_host.Resize(nnet_out.NumRows(), nnet_out.NumCols());
          nnet_out.CopyToMat(&nnet_out_host);

          assert(nnet_out_host.NumRows() == feats.NumRows());

          for(int j = 0; j < index; ++j){
             for(int k = 0; k < max_state; ++k){
                probArr[k] = nnet_out_host(j*max_state + k, 0); 
             }
             pathArr[j][ i % pathArr[j].size() ] = sample(probArr) + 1;
          }
          tot_t += feats.NumRows();

          KALDI_LOG << num_done << "\t" << i ; 

       }

       for(int j = 0; j < index; ++j)
          path_writer.Write(featKey[j], pathArr[j]);

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


