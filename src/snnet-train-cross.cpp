#include "nnet/nnet-trnopts.h"
#include "nnet/nnet-nnet.h"
#include "nnet/nnet-loss.h"
#include "nnet/nnet-randomizer.h"
#include "nnet/nnet-randomizer.cc"  // for template class
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "base/timer.h"
#include "cudamatrix/cu-device.h"
#include "util.h"
#include "snnet.h"
#include <sstream>

using namespace std;
using namespace kaldi;
using namespace kaldi::nnet1;

typedef StdVectorRandomizer<CuMatrix<BaseFloat>* > MatrixPtRandomizer;
typedef StdVectorRandomizer<vector<uchar>* >       LabelPtRandomizer;

int main(int argc, char *argv[]) {
  
  try {
    string usage;
    usage.append("Perform training cross validation.\n")
       .append("Use feature, label and path to evaluate cross entropy or mse. \n")
       .append("Usage: ").append(argv[0]).append(" [options] <feature-rspecifier> <label-rspecifier> <score-path-rspecifier> <nnet1-in> <nnet2-in> <stateMax>\n")
       .append("e.g.: \n")
       .append(" ").append(argv[0]).append(" ark:feat.ark ark:lab.ark \"ark:lattice-to-vec ark:1.lat ark:- |\" nnet.init nnet2.init 48\n");

    ParseOptions po(usage.c_str());

    bool binary = true;

    po.Register("binary", &binary, "Write model in binary mode");

    string error_function = "fer";
    po.Register("error-function", &error_function, "Error function : fer|per");

    string use_gpu="yes";
    po.Register("use-gpu", &use_gpu, "yes|no|optional, only has effect if compiled with CUDA");

    double dropout_retention = 0.0;
    po.Register("dropout-retention", &dropout_retention, "number between 0..1, saying how many neurons to preserve (0.0 will keep original value)");
     
    string feature_transform;
    po.Register("feature-transform", &feature_transform, "Feature transform in front of main network (in nnet format)");

    po.Read(argc, argv);

    if (po.NumArgs() != 6) {
      po.PrintUsage();
      exit(1);
    }

    string feat_rspecifier       = po.GetArg(1),
           label_rspecifier      = po.GetArg(2),
           score_path_rspecifier = po.GetArg(3),
           nnet1_in_filename     = po.GetArg(4),
           nnet2_in_filename     = po.GetArg(5);
    int    stateMax              = atoi(po.GetArg(6).c_str());

    // function pointer used in calculating target.
    double (*acc_function)(const vector<uchar>& path1, const vector<uchar>& path2, double param);

    if(error_function == "fer")
       acc_function = frame_acc;
    else if(error_function == "per")
       acc_function = phone_acc; 
    else{
       po.PrintUsage();
       exit(1);
    }

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

    KALDI_LOG << "SNNet cross validation started.";

    Timer time;
    int32 num_done = 0;

    SequentialScorePathReader        score_path_reader(score_path_rspecifier);
    SequentialBaseFloatMatrixReader  feature_reader(feat_rspecifier);
    SequentialUcharVectorReader      label_reader(label_rspecifier);

    Strt strt;

    CuMatrix<BaseFloat> obj_diff;
    CuMatrix<BaseFloat> nnet_out;
    CuMatrix<BaseFloat> nnet_tgt_device;
    for ( ; !(score_path_reader.Done() || feature_reader.Done() || label_reader.Done());
          score_path_reader.Next(), feature_reader.Next(), label_reader.Next(), num_done++) {

       assert( score_path_reader.Key() == feature_reader.Key() );
       assert( label_reader.Key() == feature_reader.Key() );

#if HAVE_CUDA==1
       // check the GPU is not overheated
       CuDevice::Instantiate().CheckGpuHealth();
#endif
       CuMatrix<BaseFloat>       feat(feature_reader.Value());
       ScorePath::Table          table = score_path_reader.Value().Value();
       vector<uchar>             label = label_reader.Value();

       vector<vector<uchar> * >  nnet_label_in(table.size() + 1);
       Vector< BaseFloat >       delta(table.size() + 1);

       for(int i = 0; i < table.size(); ++i){
          nnet_label_in[i]    = &table[i].second;
          delta(i)            = (1 - acc_function(label, table[i].second, 1.0)); // error rate
       }
       nnet_label_in[table.size()] = &label;

       nnet.Feedforward(feat, nnet_label_in, &nnet_out);

       // f(x, y) - f(x, y_hat)
       nnet_out.Add(-nnet_out(table.size(), 0));

       strt.Eval(delta, nnet_out, NULL);
    } 

    KALDI_LOG << "Done " << num_done << " examples, " 
              << " with other errors. "
              << " " << time.Elapsed()/60 << " min";

    KALDI_LOG << strt.Report();

#if HAVE_CUDA==1
    CuDevice::Instantiate().PrintProfile();
#endif

    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

