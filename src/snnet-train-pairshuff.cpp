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
typedef StdVectorRandomizer<int>                 TypeRandomizer;

int main(int argc, char *argv[]) {
  
  try {
    string usage;
    usage.append("Perform one iteration of Structure Neural Network training by shuffle mini-batch Stochastic Gradient Descent on training pairs.\n")
       .append("Use feature, label and path to train the neural net. \n")
       .append("Usage: ").append(argv[0]).append(" [options] <feature-rspecifier> <label-rspecifier> <score-path-rspecifier> <nnet1-in> <nnet2-in> <stateMax> [<nnet1-out> <nnet2-out>]\n")
       .append("e.g.: \n")
       .append(" ").append(argv[0]).append(" ark:feat.ark ark:lab.ark \"ark:lattice-to-vec ark:1.lat ark:- |\" nnet.init nnet2.init 48 nnet.iter1 nnet2.iter1\n");

    ParseOptions po(usage.c_str());

    NnetTrainOptions trn_opts, trn_opts_tmp;
    trn_opts.Register(&po);
    NnetDataRandomizerOptions rnd_opts;
    rnd_opts.Register(&po);

    bool binary = true, 
         crossvalidate = false,
         randomize = true;

    po.Register("binary", &binary, "Write model in binary mode");
    po.Register("cross-validate", &crossvalidate, "Perform cross-validation (don't backpropagate)");
    po.Register("randomize", &randomize, "Perform the frame-level shuffling within the Cache");

    string error_function = "fer";
    po.Register("error-function", &error_function, "Error function : fer|per");

    string use_gpu="yes";
    po.Register("use-gpu", &use_gpu, "yes|no|optional, only has effect if compiled with CUDA");

    double dropout_retention = 0.0;
    po.Register("dropout-retention", &dropout_retention, "number between 0..1, saying how many neurons to preserve (0.0 will keep original value)");
     
    int negative_num = 0;
    po.Register("negative-num", &negative_num, "insert negative example in training");

    double nnet_ratio = 1.0;
    po.Register("nnet-ratio", &nnet_ratio, "nnet1 learning rate ratio");

    string feature_transform;
    po.Register("feature-transform", &feature_transform, "Feature transform in front of main network (in nnet format)");

    bool binary_error = false;
    po.Register("binary-error", &binary_error, "Use binary error ( 1, -1 ) instead of real number");

    po.Read(argc, argv);

    if (po.NumArgs() != 8-(crossvalidate?2:0)) {
      po.PrintUsage();
      exit(1);
    }


    string feat_rspecifier       = po.GetArg(1),
           label_rspecifier      = po.GetArg(2),
           score_path_rspecifier = po.GetArg(3),
           nnet1_in_filename     = po.GetArg(4),
           nnet2_in_filename     = po.GetArg(5);
    int    stateMax              = atoi(po.GetArg(6).c_str());

    string nnet1_out_filename, nnet2_out_filename;

    if(!crossvalidate){
       nnet1_out_filename = po.GetArg(7);
       nnet2_out_filename = po.GetArg(8);
    }

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
    // ------------------------------------------------------------
    // read in all features and save to GPU
    // read in all labels and save to CPU
    SequentialScorePathReader        score_path_reader(score_path_rspecifier);
    SequentialBaseFloatMatrixReader  feature_reader(feat_rspecifier);
    SequentialUcharVectorReader      label_reader(label_rspecifier);

    vector< CuMatrix<BaseFloat> >     features; // all features
    vector< vector< uchar > >         labels;   // reference labels
    vector< vector< vector<uchar> > > examples; // including positive & negative examples
    vector< vector< int > >           types;

    features.reserve(BUFSIZE);
    labels.reserve(BUFSIZE);
    examples.reserve(BUFSIZE);
    types.reserve(BUFSIZE);

    srand(rnd_opts.randomizer_seed);

    for ( ; !(score_path_reader.Done() || feature_reader.Done() || label_reader.Done());
          score_path_reader.Next(), feature_reader.Next(), label_reader.Next()) {

       assert( score_path_reader.Key() == feature_reader.Key() );
       assert( label_reader.Key() == feature_reader.Key() );

       const ScorePath::Table  &table = score_path_reader.Value().Value();
       const Matrix<BaseFloat> &feat  = feature_reader.Value();
       const vector<uchar>     &label = label_reader.Value();

       features.push_back(CuMatrix<BaseFloat>(feat));
       labels.push_back(label);
       examples.push_back(vector< vector<uchar> >());
       types.push_back(vector< int > ());

       vector< vector<uchar> > &seqs = examples[examples.size() - 1];
       vector< int >           &tps  = types[types.size() - 1];

       //seqs.reserve(table.size() + negative_num);
       //tps.reserve(table.size() + negative_num);

       seqs.reserve(1 + table.size() + negative_num);
       tps.reserve(1 + table.size() + negative_num);

       // positive examples
       seqs.push_back(label);
       tps.push_back(REF_TYPE);

       // negative examples
       for(int i = 0; i < table.size(); ++i){
          seqs.push_back(table[i].second);
          tps.push_back(LAT_TYPE);
       }

       // TODO set random seed
       // random negitive examples
       vector<uchar> neg_arr(label.size());
       for(int i = 0; i < negative_num; ++i){
          for(int j = 0; j < neg_arr.size(); ++j)
             neg_arr[j] = rand() % stateMax + 1;
          seqs.push_back(neg_arr);
          tps.push_back(RAND_TYPE);
       }
    } 
    // -------------------------------------------------------------

    int numTotal = 0;
    RandomizerMask       randomizer_mask(rnd_opts);

    MatrixPtRandomizer   feature_randomizer(rnd_opts);
    LabelPtRandomizer    label_randomizer(rnd_opts);
    LabelPtRandomizer    ref_label_randomizer(rnd_opts);
    VectorRandomizer     delta_randomizer(rnd_opts);
    TypeRandomizer       type_randomizer(rnd_opts);
    
    KALDI_LOG << "Filling all randomizer. features # = " << features.size();
    KALDI_LOG << " each features get " << examples[0].size() << " exs.";
    // fill all data into randomizer
    for(int i = 0; i < examples.size(); ++i){

       KALDI_ASSERT (!feature_randomizer.IsFull());

       int total = examples[i].size()*(examples[i].size()-1)/2;

       vector< CuMatrix<BaseFloat>* > feat(total);
       vector< vector<uchar>* >       lab(total);
       vector< vector<uchar>* >       ref_lab(total);
       Vector< BaseFloat >            dlt(total, kSetZero); 
       vector< int >                  tps(total);

       int j = 0;
       for(int m = 0; m < examples[i].size(); ++m){
          for(int n = m + 1; n < examples[i].size(); ++n){
             feat[j]    = &features[i];
             lab[j]     = &examples[i][m];
             ref_lab[j] = &examples[i][n];
             dlt(j)     = acc_function(labels[i], examples[i][m], 1.0) -
                          acc_function(labels[i], examples[i][n], 1.0) ; // error rate
             tps[j]      = types[i][m] * END_TYPE + types[i][n];

             if(dlt(j) != 0.0) j++;
          }
       }
       feat.resize(j);
       lab.resize(j);
       ref_lab.resize(j);
       dlt.Resize(j, kCopyData);
       tps.resize(j);

       numTotal += j;

       feature_randomizer.AddData(feat);
       label_randomizer.AddData(lab);
       ref_label_randomizer.AddData(ref_lab);
       delta_randomizer.AddData(dlt);
       type_randomizer.AddData(tps);
    }
    KALDI_LOG << "Filled all data. (" << numTotal << ")";

    // prepare Nnet
    SNnet nnet;
    nnet.Read(nnet1_in_filename, nnet2_in_filename, stateMax);
    nnet.SetTrainOptions(trn_opts, nnet_ratio);

    if (feature_transform != "") {
       Nnet nnet_transf;
       nnet_transf.Read(feature_transform);
       nnet.SetTransform(nnet_transf);
    }

    if (dropout_retention > 0.0) {
      nnet.SetDropoutRetention(dropout_retention);
    }

    if (crossvalidate) {
      nnet.SetDropoutRetention(1.0);
    }

    StrtCmp strt(binary_error);

    Timer time;
    KALDI_LOG << (crossvalidate?"CROSS-VALIDATION":"TRAINING") << " STARTED";

    // randomize
    if (!crossvalidate && randomize) {
       const std::vector<int32>& mask = randomizer_mask.Generate(feature_randomizer.NumFrames());

       feature_randomizer.Randomize(mask);
       label_randomizer.Randomize(mask);
       ref_label_randomizer.Randomize(mask);
       delta_randomizer.Randomize(mask);
       type_randomizer.Randomize(mask);
    }

    int64 num_done = 0;
    CuMatrix<BaseFloat>          nnet_out;
    vector<CuMatrix<BaseFloat> > obj_diff(2);
    // train with data from randomizers (using mini-batches)
    for ( ; !feature_randomizer.Done(); feature_randomizer.Next(), label_randomizer.Next(),
          ref_label_randomizer.Next(), delta_randomizer.Next(), type_randomizer.Next()){

#if HAVE_CUDA==1
       // check the GPU is not overheated
       CuDevice::Instantiate().CheckGpuHealth();
#endif

       // get block of feature/delta pairs
       const vector<CuMatrix<BaseFloat>* > &nnet_feat_in   = feature_randomizer.Value();
       const vector<vector<uchar> * >      &nnet_label_in  = label_randomizer.Value();
       const vector<vector<uchar> * >      &nnet_ref_label = ref_label_randomizer.Value();
       const Vector<BaseFloat>             &delta          = delta_randomizer.Value();

       const vector<int>                   &example_type   = type_randomizer.Value();

       // Forward pass
       // nnet_out = f(x, y) - f(x, y_hat)
       nnet.Propagate(nnet_feat_in, nnet_label_in, nnet_ref_label, &nnet_out);

       int counter = 0;
       strt.Eval(delta, nnet_out, &obj_diff, &counter, &example_type);

       trn_opts_tmp = trn_opts;
       trn_opts_tmp.learn_rate *= counter;
       nnet.SetTrainOptions(trn_opts_tmp, nnet_ratio);


       // backward pass
       if (!crossvalidate && counter != 0) {
          // backpropagate
          nnet.Backpropagate(obj_diff, nnet_label_in, nnet_ref_label);
       }

       num_done += nnet_feat_in.size();
    }

    // after last minibatch : show what happens in network 
    if (kaldi::g_kaldi_verbose_level >= 1) { // vlog-1
       KALDI_VLOG(1) << "###############################";
       KALDI_VLOG(1) << nnet.InfoPropagate();
       if (!crossvalidate) {
          KALDI_VLOG(1) << nnet.InfoBackPropagate();
          KALDI_VLOG(1) << nnet.InfoGradient();
       }
    }


    if (!crossvalidate) {
       nnet.Write(nnet1_out_filename, nnet2_out_filename, binary);
    }

    KALDI_LOG << "Done " << num_done << " examples, " 
              << " with other errors. "
              << "[" << (crossvalidate?"CROSS-VALIDATION":"TRAINING")
              << ", " << (randomize?"RANDOMIZED":"NOT-RANDOMIZED") 
              << ", " << time.Elapsed()/60 << " min"
              << "]";  

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

