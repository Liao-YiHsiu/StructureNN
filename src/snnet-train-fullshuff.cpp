#include "nnet/nnet-trnopts.h"
#include "nnet/nnet-nnet.h"
#include "nnet/nnet-loss.h"
#include "nnet/nnet-randomizer.h"
#include "nnet/nnet-randomizer.cc"  // for template class
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "base/timer.h"
#include "cudamatrix/cu-device.h"
#include "svm.h"
#include "snnet.h"
#include <sstream>

#define BUFSIZE 4096

using namespace std;
using namespace kaldi;
using namespace kaldi::nnet1;

typedef StdVectorRandomizer<CuMatrix<BaseFloat>* > MatrixPtRandomizer;
typedef StdVectorRandomizer<vector<uchar>* >       LabelPtRandomizer;

int main(int argc, char *argv[]) {
  
  try {
    string usage;
    usage.append("Perform one iteration of Structure Neural Network training by full shuffle mini-batch Stochastic Gradient Descent.\n")
       .append("Use feature, label and path to train the neural net. \n")
       .append("Usage: ").append(argv[0]).append(" [options] <feature-rspecifier> <label-rspecifier> <score-path-rspecifier> <nnet1-in> <nnet2-in> <stateMax> [<nnet1-out> <nnet2-out>]\n")
       .append("e.g.: \n")
       .append(" ").append(argv[0]).append(" ark:feat.ark ark:lab.ark \"ark:lattice-to-vec ark:1.lat ark:- |\" nnet.init nnet2.init 48 nnet.iter1 nnet2.iter1\n");

    ParseOptions po(usage.c_str());

    NnetTrainOptions trn_opts;
    trn_opts.Register(&po);
    NnetDataRandomizerOptions rnd_opts;
    rnd_opts.Register(&po);

    bool binary = true, 
         crossvalidate = false,
         randomize = true;

    po.Register("binary", &binary, "Write model in binary mode");
    po.Register("cross-validate", &crossvalidate, "Perform cross-validation (don't backpropagate)");
    po.Register("randomize", &randomize, "Perform the frame-level shuffling within the Cache");

    string objective_function = "xent";
    po.Register("objective-function", &objective_function, "Objective function : xent|mse");

    string error_function = "fer";
    po.Register("error-function", &error_function, "Error function : fer|per");

    string use_gpu="yes";
    po.Register("use-gpu", &use_gpu, "yes|no|optional, only has effect if compiled with CUDA");

    double dropout_retention = 0.0;
    po.Register("dropout-retention", &dropout_retention, "number between 0..1, saying how many neurons to preserve (0.0 will keep original value)");
     
    int negative_num = 0;
    po.Register("negative-num", &negative_num, "insert negative example in training");

    bool reweight = false;
    po.Register("reweight", &reweight, "reweight training examles");

    string feature_transform;
    po.Register("feature-transform", &feature_transform, "Feature transform in front of main network (in nnet format)");

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
    vector< vector< BaseFloat > >     weights;

    features.reserve(BUFSIZE);
    labels.reserve(BUFSIZE);
    examples.reserve(BUFSIZE);
    weights.reserve(BUFSIZE);

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
       weights.push_back(vector<BaseFloat>());

       vector< vector<uchar> > &seqs = examples[examples.size() - 1];
       vector< BaseFloat > &wght = weights[weights.size() - 1];

       seqs.reserve(1 + table.size() + negative_num);
       wght.reserve(1 + table.size() + negative_num);

       // positive examples
       seqs.push_back(label);
       wght.push_back(1.0);

       // negative examples
       for(int i = 0; i < table.size(); ++i){
          seqs.push_back(table[i].second);
          wght.push_back(1.0);
       }

       // TODO set random seed
       // random negitive examples
       vector<uchar> neg_arr(label.size());
       for(int i = 0; i < negative_num; ++i){
          for(int j = 0; j < neg_arr.size(); ++j)
             neg_arr[j] = rand() % stateMax + 1;
          seqs.push_back(neg_arr);
          wght.push_back(1.0);
       }
    } 
    // -------------------------------------------------------------

    int numTotal = 0;
    RandomizerMask       randomizer_mask(rnd_opts);
    MatrixPtRandomizer   feature_randomizer(rnd_opts);
    LabelPtRandomizer    label_randomizer(rnd_opts);
    VectorRandomizer      target_randomizer(rnd_opts);
    VectorRandomizer      weights_randomizer(rnd_opts);
    
    KALDI_LOG << "Filling all randomizer. features # = " << features.size();
    KALDI_LOG << " each features get " << examples[0].size() << " exs.";
    // fill all data into randomizer
    for(int i = 0; i < features.size(); ++i){
       vector< CuMatrix<BaseFloat>* > feat(examples[i].size());
       vector< vector<uchar>* >       lab(examples[i].size());
       Vector< BaseFloat >            tgt(examples[i].size(), kSetZero); 
       Vector< BaseFloat >            wgt(examples[i].size(), kSetZero);

       for(int j = 0; j < examples[i].size(); ++j){
          feat[j] = &features[i];
          lab[j]  = &examples[i][j];
          tgt(j)  = acc_function(labels[i], examples[i][j], 1.0);
          wgt(j)  = weights[i][j];
       }

       numTotal += examples[i].size();

       feature_randomizer.AddData(feat);
       label_randomizer.AddData(lab);
       target_randomizer.AddData(tgt);
       weights_randomizer.AddData(wgt);
    }
    KALDI_LOG << "Filled all data.";

    // prepare Nnet
    SNnet nnet;
    nnet.Read(nnet1_in_filename, nnet2_in_filename, stateMax);
    nnet.SetTrainOptions(trn_opts);

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

    Xent xent;
    Mse mse;

    Timer time;
    KALDI_LOG << (crossvalidate?"CROSS-VALIDATION":"TRAINING") << " STARTED";

    // randomize
    if (!crossvalidate && randomize) {
       const std::vector<int32>& mask = randomizer_mask.Generate(feature_randomizer.NumFrames());
       feature_randomizer.Randomize(mask);
       label_randomizer.Randomize(mask);
       target_randomizer.Randomize(mask);
       weights_randomizer.Randomize(mask);
    }

    int64 num_done = 0;
    CuMatrix<BaseFloat> nnet_out;
    CuMatrix<BaseFloat> obj_diff;
    CuMatrix<BaseFloat> nnet_tgt_device;
    Matrix<BaseFloat>   nnet_tgt_host;
    // train with data from randomizers (using mini-batches)
    for ( ; !feature_randomizer.Done(); feature_randomizer.Next(),
          label_randomizer.Next(), target_randomizer.Next()){

#if HAVE_CUDA==1
       // check the GPU is not overheated
       CuDevice::Instantiate().CheckGpuHealth();
#endif

       // get block of feature/target pairs
       const vector<CuMatrix<BaseFloat>* > &nnet_feat_in = feature_randomizer.Value();
       const vector<vector<uchar> * > &nnet_label_in = label_randomizer.Value();
       const Vector<BaseFloat> &nnet_tgt = target_randomizer.Value();
       const Vector<BaseFloat> &frm_weights = weights_randomizer.Value();

       nnet_tgt_host.Resize(nnet_tgt.Dim(), 1, kSetZero);
       nnet_tgt_host.CopyColsFromVec(nnet_tgt);

       nnet_tgt_device = nnet_tgt_host;

       //nnet_tgt_device.Resize(nnet_tgt.Dim(), 1, kSetZero);
       //nnet_tgt_device.CopyColsFromVec(nnet_tgt);
       //CuSubVector<BaseFloat> tgt_row = nnet_tgt_device.Row(0);
       //tgt_row.CopyFromVec(nnet_tgt);

       // Forward pass
       nnet.Propagate(nnet_feat_in, nnet_label_in, &nnet_out);

       // evaluate objective function we've chosen
       if (objective_function == "xent") {
          xent.Eval(frm_weights, nnet_out, nnet_tgt_device, &obj_diff); 
       } else if (objective_function == "mse") {
          mse.Eval(frm_weights, nnet_out, nnet_tgt_device, &obj_diff);
       } else {
          KALDI_ERR << "Unknown objective function code : " << objective_function;
       }

       // backward pass
       if (!crossvalidate) {
          // backpropagate
          nnet.Backpropagate(obj_diff);
       }

       num_done += nnet_feat_in.size();
       KALDI_LOG << "Done: " << num_done << "/" << numTotal;
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

    if (objective_function == "xent") {
      KALDI_LOG << xent.Report();
    } else if (objective_function == "mse") {
      KALDI_LOG << mse.Report();
    } else {
      KALDI_ERR << "Unknown objective function code : " << objective_function;
    }

#if HAVE_CUDA==1
    CuDevice::Instantiate().PrintProfile();
#endif

    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

