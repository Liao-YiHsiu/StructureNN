#include "nnet/nnet-trnopts.h"
#include "nnet/nnet-nnet.h"
#include "nnet/nnet-loss.h"
#include "nnet/nnet-randomizer.h"
#include "nnet/nnet-randomizer.cc" 
#include "nnet/nnet-various.h"
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
typedef StdVectorRandomizer<int>                 TypeRandomizer;

int main(int argc, char *argv[]) {
  
  try {
    string usage;
    usage.append("Propagate half way of Structure Neural Network to generate Psi.\n")
       .append("Use feature, label and path and front end DNN to generate Psi.\n")
       .append("Usage: ").append(argv[0]).append(" [options] <feature-rspecifier> <label-rspecifier> <score-path-rspecifier> <nnet1-in> <stateMax> <psi-wspecifier>\n")
       .append("e.g.: \n")
       .append(" ").append(argv[0]).append(" ark:feat.ark ark:lab.ark \"ark:lattice-to-vec ark:1.lat ark:- |\" nnet1.init 48 ark:psi.ark\n");

    ParseOptions po(usage.c_str());

    bool binary = true;
    po.Register("binary", &binary, "Write model in binary mode");

    string use_gpu="yes";
    po.Register("use-gpu", &use_gpu, "yes|no|optional, only has effect if compiled with CUDA");

    double dropout_retention = 0.0;
    po.Register("dropout-retention", &dropout_retention, "number between 0..1, saying how many neurons to preserve (0.0 will keep original value)");
     
    int negative_num = 0;
    po.Register("negative-num", &negative_num, "insert negative example in training");

    int rand_seed = 777;
    po.Register("rand-seed", &rand_seed, "random seed for negative example");

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
           nnet1_in_filename     = po.GetArg(4);
    int    stateMax              = atoi(po.GetArg(5).c_str());
    string psi_wspecifier        = po.GetArg(6);

    //Select the GPU
#if HAVE_CUDA==1
    CuDevice::Instantiate().SelectGpuId(use_gpu);
#endif

    // ------------------------------------------------------------
    // read in all features and save to GPU
    // read in all labels and save to CPU
    SequentialScorePathReader         score_path_reader(score_path_rspecifier);
    SequentialBaseFloatMatrixReader   feature_reader(feat_rspecifier);
    SequentialUcharVectorReader       label_reader(label_rspecifier);
    BaseFloatMatrixWriter             psi_writer(psi_wspecifier);

    vector< CuMatrix<BaseFloat> >     features; // all features
    vector< vector< vector<uchar> > > examples; // including positive & negative examples
    vector< string >                  features_name;

    srand(rand_seed);

    Timer time;
    int64 num_done = 0;

    KALDI_LOG << "Filling data";

    for ( ; !(score_path_reader.Done() || feature_reader.Done() || label_reader.Done());
          score_path_reader.Next(), feature_reader.Next(), label_reader.Next()) {


       assert( score_path_reader.Key() == feature_reader.Key() );
       assert( label_reader.Key() == feature_reader.Key() );

       const ScorePath::Table  &table = score_path_reader.Value().Value();
       const Matrix<BaseFloat> &feat  = feature_reader.Value();
       const vector<uchar>     &label = label_reader.Value();

       features.push_back(CuMatrix<BaseFloat>(feat));
       examples.push_back(vector< vector<uchar> >());
       features_name.push_back( score_path_reader.Key());

       vector< vector<uchar> > &seqs = examples[examples.size() - 1];

       seqs.reserve(1 + table.size() + negative_num);

       // positive examples
       seqs.push_back(label);

       // negative examples
       for(int i = 0; i < table.size(); ++i){
          seqs.push_back(table[i].second);
       }

       // random negitive examples
       vector<uchar> neg_arr(label.size());
       for(int i = 0; i < negative_num; ++i){
          for(int j = 0; j < neg_arr.size(); ++j)
             neg_arr[j] = rand() % stateMax + 1;
          seqs.push_back(neg_arr);
       }

       num_done += seqs.size();
    } 

    KALDI_LOG << "Filled all data. features # = " << features.size() 
       << " (" << num_done << ")";


    SNnet nnet;
    nnet.Read(nnet1_in_filename, stateMax);

    if (feature_transform != "") {
       Nnet nnet_transf;
       nnet_transf.Read(feature_transform);
       nnet.SetTransform(nnet_transf);
    }

    if (dropout_retention > 0.0) {
      nnet.SetDropoutRetention(dropout_retention);
    }


    CuMatrix<BaseFloat> nnet_out;
    Matrix<BaseFloat>   nnet_out_host;
    for(int i = 0; i < examples.size(); ++i){
#if HAVE_CUDA==1
       // check the GPU is not overheated
       CuDevice::Instantiate().CheckGpuHealth();
#endif

       vector< vector<uchar>* > labels(examples[i].size());
       for(int j =0 ; j < labels.size(); ++j)
          labels[j] = &examples[i][j];

       nnet.PropagatePsi(features[i], labels, &nnet_out);

       nnet_out_host.Resize(nnet_out.NumRows(), nnet_out.NumCols());
       nnet_out_host.CopyFromMat(nnet_out);
       psi_writer.Write(features_name[i], nnet_out_host);

    }

    KALDI_LOG << "Propagate Finished.";

    KALDI_LOG << "Done " << num_done << " examples, " 
              << " with other errors. "
              << "[ " << time.Elapsed()/60 << " min"
              << "]";  

#if HAVE_CUDA==1
    CuDevice::Instantiate().PrintProfile();
#endif

    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

