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
    usage.append("Calculate CMVN of Psi in Structure Neural Network by passing training dataset.\n")
       .append("Use feature, label and path to accumulate cmvn parameters and output as nnet. \n")
       .append("Usage: ").append(argv[0]).append(" [options] <feature-rspecifier> <label-rspecifier> <score-path-rspecifier> <nnet1-in> <nnet2-in> <stateMax> <nnet-out>\n")
       .append("e.g.: \n")
       .append(" ").append(argv[0]).append(" ark:feat.ark ark:lab.ark \"ark:lattice-to-vec ark:1.lat ark:- |\" nnet1.init 48 nnet.cvmn\n");

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

    if (po.NumArgs() != 7) {
      po.PrintUsage();
      exit(1);
    }

    string feat_rspecifier       = po.GetArg(1),
           label_rspecifier      = po.GetArg(2),
           score_path_rspecifier = po.GetArg(3),
           nnet1_in_filename     = po.GetArg(4),
           nnet2_in_filename     = po.GetArg(5);
    int    stateMax              = atoi(po.GetArg(6).c_str());
    string nnet_out_filename     = po.GetArg(7);

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

    vector< CuMatrix<BaseFloat> >     features; // all features
    vector< vector< vector<uchar> > > examples; // including positive & negative examples

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
    nnet.Read(nnet1_in_filename, nnet2_in_filename, stateMax);

    if (feature_transform != "") {
       Nnet nnet_transf;
       nnet_transf.Read(feature_transform);
       nnet.SetTransform(nnet_transf);
    }

    if (dropout_retention > 0.0) {
      nnet.SetDropoutRetention(dropout_retention);
    }


    for(int i = 0; i < examples.size(); ++i){
#if HAVE_CUDA==1
       // check the GPU is not overheated
       CuDevice::Instantiate().CheckGpuHealth();
#endif

       vector< vector<uchar>* > labels(examples[i].size());
       for(int j =0 ; j < labels.size(); ++j)
          labels[j] = &examples[i][j];

       nnet.Acc(features[i], labels);
    }

    KALDI_LOG << "Propagate Finished.";

    // -------------------------------------------------------------
    CuVector<BaseFloat> mean, sd;
    nnet.Stat(mean, sd);

    mean.Scale(-1);

    sd.Add(1e-5);
    sd.ApplyPow(-1);

    AddShift *addshift = new AddShift(mean.Dim(), mean.Dim());
    addshift->SetShiftVec(mean);
    addshift->SetLearnRateCoef(0);

    Rescale *rescale = new Rescale(mean.Dim(), mean.Dim());
    rescale->SetScaleVec(sd);
    rescale->SetLearnRateCoef(0);

    Nnet nnet_out;
    nnet_out.AppendComponent(addshift);
    nnet_out.AppendComponent(rescale);
    nnet_out.Write(nnet_out_filename, binary);

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

