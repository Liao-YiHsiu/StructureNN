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
#include <omp.h>

using namespace std;
using namespace kaldi;
using namespace kaldi::nnet1;

typedef StdVectorRandomizer<CuMatrix<BaseFloat>* >    MatrixPtRandomizer;
typedef StdVectorRandomizer<vector<vector<uchar> >* > LabelArrPtRandomizer;
typedef StdVectorRandomizer<vector<BaseFloat>* >      TargetArrPtRandomizer;

int main(int argc, char *argv[]) {
  
  try {
    string usage;
    usage.append("Perform one iteration of Structure Neural Network training by shuffle mini-batch Stochastic Gradient Descent on training lists. Use learning to rank techniques. \n")
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

    string acc_func = "fac";
    po.Register("acc-func", &acc_func, "Acuracy function : fac|pac");

    bool acc_norm = true;
    po.Register("acc-norm", &acc_norm, "normalization acc function to [0, 1]");

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

    po.Read(argc, argv);

    if (po.NumArgs() != 8-(crossvalidate?2:0)) {
      po.PrintUsage();
      exit(1);
    }

    // setup input parameters

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
    double (*acc_function)(const vector<uchar>& path1, const vector<uchar>& path2, bool norm);

    if(acc_func == "fac")
       acc_function = frame_acc;
    else if(acc_func == "pac")
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
    vector< vector< vector<uchar> > > examples; // including positive & negative examples
    vector< vector< BaseFloat > >     targets;  // target value of examples

    features.reserve(BUFSIZE);
    examples.reserve(BUFSIZE);

    srand(rnd_opts.randomizer_seed);

    for ( ; !(score_path_reader.Done() || feature_reader.Done() || label_reader.Done());
          score_path_reader.Next(), feature_reader.Next(), label_reader.Next()) {

       assert( score_path_reader.Key() == feature_reader.Key() );
       assert( label_reader.Key() == feature_reader.Key() );

       const ScorePath::Table  &table = score_path_reader.Value().Value();
       const Matrix<BaseFloat> &feat  = feature_reader.Value();
       const vector<uchar>     &label = label_reader.Value();

       features.push_back(CuMatrix<BaseFloat>(feat));
       examples.push_back(vector< vector<uchar> >());
       targets.push_back(vector<BaseFloat> ());

       vector< vector<uchar> > &seqs = examples[examples.size() - 1];
       vector< BaseFloat >     &tgts = targets[targets.size() -1];

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

       tgts.resize(seqs.size());
#pragma omp parallel for
       for(int i = 0; i < seqs.size(); ++i)
          tgts[i] = acc_function(label, seqs[i], acc_norm);
    } 
    // -------------------------------------------------------------
    //
    // fixed minibatch_size = 1
    rnd_opts.minibatch_size = 1;

    RandomizerMask       randomizer_mask(rnd_opts);

    MatrixPtRandomizer    feature_randomizer(rnd_opts);
    LabelArrPtRandomizer  label_randomizer(rnd_opts);
    TargetArrPtRandomizer target_randomizer(rnd_opts);

    {
       vector< CuMatrix<BaseFloat>* >     feats;
       vector< vector< vector<uchar> >* > labs;
       vector< vector< BaseFloat >*  >    tgts;

       VecToVecRef(features, feats);
       VecToVecRef(examples, labs);
       VecToVecRef(targets, tgts);

       feature_randomizer.AddData(feats);
       label_randomizer.AddData(labs);
       target_randomizer.AddData(tgts);
    }

    KALDI_ASSERT (!feature_randomizer.IsFull());
    
    KALDI_LOG << "Filled all randomizer. features # = " << features.size();
    KALDI_LOG << " each features get " << examples[0].size() << " exs.";
    
    // randomize
    if (!crossvalidate && randomize) {
       const std::vector<int32>& mask = 
          randomizer_mask.Generate(feature_randomizer.NumFrames());

       feature_randomizer.Randomize(mask);
       label_randomizer.Randomize(mask);
       target_randomizer.Randomize(mask);
    }

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

    Timer time;

    KALDI_LOG << (crossvalidate?"CROSS-VALIDATION":"TRAINING") << " STARTED";

    int64 num_done = 0;
    CuMatrix<BaseFloat> nnet_out;
    CuMatrix<BaseFloat> obj_diff;

    StrtListBase strt;
    strt.SetAll(features.size());

    for ( ; !feature_randomizer.Done(); feature_randomizer.Next(), 
          label_randomizer.Next(), target_randomizer.Next()){

#if HAVE_CUDA==1
       // check the GPU is not overheated
       CuDevice::Instantiate().CheckGpuHealth();
#endif
       assert(feature_randomizer.Value().size() == 1 &&
             label_randomizer.Value().size() == 1 && target_randomizer.Value().size() == 1);

       const CuMatrix<BaseFloat>    &nnet_feat_in   = *(feature_randomizer.Value()[0]);
       const vector<BaseFloat>      &nnet_target    = *(target_randomizer.Value()[0]);

       vector<vector<uchar> > &nnet_label_in  = *(label_randomizer.Value()[0]);

       vector< vector<uchar>* > nnet_label_in_ref;

       VecToVecRef(nnet_label_in, nnet_label_in_ref);

       nnet.Propagate(nnet_feat_in, nnet_label_in_ref, &nnet_out);

       strt.Eval(nnet_target, nnet_out, &obj_diff);

       // backpropagate
       if (!crossvalidate) {
          nnet.Backpropagate(obj_diff, nnet_label_in_ref);
       }

       num_done += nnet_label_in.size();
    }
    KALDI_LOG << "Done " << num_done << " label sequences.";


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

