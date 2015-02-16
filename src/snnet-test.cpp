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
    usage.append("Structure Neural Network predict.\n")
       .append("Use SNN to rescore path and output the frame error rate.\n")
       .append("Usage: ").append(argv[0]).append(" [options] <feature-rspecifier> <label-rspecifier> <score-path-rspecifier> <model-in> <path-wspecifier>\n")
       .append("e.g.: \n")
       .append(" ").append(argv[0]).append(" ark:feat.ark ark:lab.ark \"ark:lattice-to-vec ark:1.lat ark:- |\" nnet ark:path.ark \n");

    ParseOptions po(usage.c_str());

    string use_gpu="yes";
    po.Register("use-gpu", &use_gpu, "yes|no|optional, only has effect if compiled with CUDA");

    int max_state = 48;
    po.Register("max-state", &max_state, "max state ID");

    po.Read(argc, argv);

    if (po.NumArgs() != 5) {
      po.PrintUsage();
      exit(1);
    }

    string feat_rspecifier  = po.GetArg(1),
      label_rspecifier      = po.GetArg(2),
      score_path_rspecifier = po.GetArg(3),
      model_filename        = po.GetArg(4),
      path_wspecifier       = po.GetArg(5);


    Int32VectorWriter                                     path_writer(path_wspecifier);
    SequentialTableReader<KaldiObjectHolder<ScorePath> >  score_path_reader(score_path_rspecifier);
    SequentialBaseFloatMatrixReader                       feature_reader(feat_rspecifier);
    SequentialInt32VectorReader                           label_reader(label_rspecifier);

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
    CuMatrix<BaseFloat> nnet_out;

   //int featsN = (labfeats[0].second[0].size() + maxState) * maxState;

    KALDI_LOG << "PREDICT STARTED";

    Timer time;
    double time_now = 0;

    int32 corr = 0;
    int32 corrN = 0;

    kaldi::int64 tot_t = 0;
    int32 num_done = 0;
    int32 featsN = -1;

    for ( ; !score_path_reader.Done();
          score_path_reader.Next(), feature_reader.Next(), label_reader.Next()) {

       //for(int i = 0; i < all_score_path.size(); ++i){
#if HAVE_CUDA==1
       // check the GPU is not overheated
       CuDevice::Instantiate().CheckGpuHealth();
#endif
       assert( score_path_reader.Key() == feature_reader.Key() );
       assert( label_reader.Key() == feature_reader.Key() );

       const ScorePath::Table  &table = score_path_reader.Value().Value();
       const Matrix<BaseFloat> &feat  = feature_reader.Value();
       const vector<int32>     &label = label_reader.Value();

       if(featsN < 0)
          featsN = (feat.NumCols() + max_state) * max_state;
       assert((feat.NumCols() + max_state)*max_state == featsN);

       Matrix<BaseFloat> feats(table.size(), featsN);
       Posterior         targets;

       for(int i = 0; i < table.size(); ++i){
          makeFeature(feat, table[i].second, max_state, feats.Row(i));
       }

       nnet.Feedforward(CuMatrix<BaseFloat>(feats), &nnet_out);

       //download from GPU
       nnet_out_host.Resize(nnet_out.NumRows(), nnet_out.NumCols());
       nnet_out.CopyToMat(&nnet_out_host);

       // find max
       BaseFloat max = -1;
       int imax;
       for(int j = 0; j < nnet_out_host.NumRows(); ++j){
          BaseFloat value = nnet_out_host(j, 0); //TODO + weight * table[j].first;
          if(value > max){
             max = value;
             imax = j;
          }
       }

       path_writer.Write(feature_reader.Key(), table[imax].second);

       corr += path_acc(table[imax].second, label) * label.size();
       corrN += label.size();

       // progress log
       if (num_done % 100 == 0) {
          time_now = time.Elapsed();
          KALDI_VLOG(1) << "After " << num_done << " utterances: time elapsed = "
             << time_now/60 << " min; processed " << tot_t/time_now
             << " frames per second.";
       }
       num_done++;
       tot_t += feats.NumRows();

    }

    // final message
    KALDI_LOG << "Done " << num_done << " files" 
              << " in " << time.Elapsed()/60 << "min," 
              << " (fps " << tot_t/time.Elapsed() << ")"; 

    KALDI_LOG << "Total FER = " << (1 - (corr/ (double)corrN));


#if HAVE_CUDA==1
    CuDevice::Instantiate().PrintProfile();
#endif

    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


