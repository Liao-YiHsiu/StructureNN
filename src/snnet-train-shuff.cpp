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

typedef StdVectorRandomizer< CuMatrix<BaseFloat>* >      MatrixPtRandomizer;
typedef StdVectorRandomizer< vector<uchar>* >            LabelPtRandomizer;
typedef StdVectorRandomizer< vector< vector<uchar>* >* > LabelArrPtRandomizer;
typedef StdVectorRandomizer< Vector< BaseFloat     >* >  ScoreArrPtRandomizer;

int main(int argc, char *argv[]) {
  
  try {
    string usage;
    usage.append("Perform one iteration of Structure Neural Network training by shuffle mini-batch Stochastic Gradient Descent (use find most violated constraint) or max instead of all.\n")
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

    features.reserve(BUFSIZE);
    labels.reserve(BUFSIZE);
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
       labels.push_back(label);
       examples.push_back(vector< vector<uchar> >());

       vector< vector<uchar> > &seqs = examples[examples.size() - 1];

       seqs.reserve(1 + table.size() + negative_num);

       // positive examples
       seqs.push_back(label);

       // negative examples
       for(int i = 0; i < table.size(); ++i){
          seqs.push_back(table[i].second);
       }

       // TODO set random seed
       // random negitive examples
       vector<uchar> neg_arr(label.size());
       for(int i = 0; i < negative_num; ++i){
          for(int j = 0; j < neg_arr.size(); ++j)
             neg_arr[j] = rand() % stateMax + 1;
          seqs.push_back(neg_arr);
       }
    } 
    
    // compute deltas
    vector< Vector< BaseFloat > >     deltas(examples.size());
    for(int i = 0; i < examples.size(); ++i){
       deltas[i].Resize(examples[i].size());
       for(int j = 0; j < examples[i].size(); ++j)
          deltas[i](j) = 1 - acc_function(labels[i], examples[i][j], 1.0);
    }

    vector< vector< vector<uchar>* > > examples_ptr(examples.size());
    for(int i = 0; i < examples.size(); ++i){
       examples_ptr[i].resize(examples[i].size());
       for(int j = 0; j < examples[i].size(); ++j)
          examples_ptr[i][j] = &examples[i][j];
    }

    // -------------------------------------------------------------
    // handle minibatch by myself
    int minibatch_size = rnd_opts.minibatch_size;
    rnd_opts.minibatch_size = 1;

    RandomizerMask       randomizer_mask(rnd_opts);
    MatrixPtRandomizer   feature_randomizer(rnd_opts);
    LabelArrPtRandomizer label_randomizer(rnd_opts);
    ScoreArrPtRandomizer delta_randomizer(rnd_opts);
    LabelPtRandomizer    ref_label_randomizer(rnd_opts);
    
    KALDI_LOG << "Filling all randomizer. features # = " << features.size();
    KALDI_LOG << " each features get " << examples[0].size() << " exs.";
    // fill all data into randomizer
    {
       vector< CuMatrix<BaseFloat>* >       feat(features.size());
       vector< vector<uchar>* >             ref_lab(features.size());
       vector< vector< vector<uchar>* >* >  lab(features.size());
       vector< Vector< BaseFloat >* >       dlt(features.size());

       for(int i = 0; i < features.size(); ++i){
          feat[i]    = &features[i];
          lab[i]     = &examples_ptr[i];
          ref_lab[i] = &labels[i];
          dlt[i]     = &deltas[i];
       }

       feature_randomizer.AddData(feat);
       label_randomizer.AddData(lab);
       ref_label_randomizer.AddData(ref_lab);
       delta_randomizer.AddData(dlt);
    }
    KALDI_LOG << "Filled all data.";

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

    Strt strt;

    Timer time;
    KALDI_LOG << (crossvalidate?"CROSS-VALIDATION":"TRAINING") << " STARTED";

    // randomize
    if (!crossvalidate && randomize) {
       const std::vector<int32>& mask = randomizer_mask.Generate(feature_randomizer.NumFrames());

       feature_randomizer.Randomize(mask);
       label_randomizer.Randomize(mask);
       ref_label_randomizer.Randomize(mask);
       delta_randomizer.Randomize(mask);
    }

    int64 num_done = 0;
    CuMatrix<BaseFloat>          nnet_out, picked_nnet_out;
    vector<CuMatrix<BaseFloat> > obj_diff(2);
    // train with data from randomizers (using mini-batches)
    while(!feature_randomizer.Done()){
       vector< CuMatrix<BaseFloat>* > picked_nnet_feat_in(minibatch_size);
       vector< vector<uchar>*       > picked_nnet_label_in(minibatch_size);
       vector< vector<uchar>*       > picked_nnet_ref_label(minibatch_size);
       vector< BaseFloat            > picked_delta(minibatch_size);

#if HAVE_CUDA==1
       // check the GPU is not overheated
       CuDevice::Instantiate().CheckGpuHealth();
#endif
       int counter = 0;

       // pick the one with max_y{f(x,y) + delta(y_hat, y)}
       for ( ; !feature_randomizer.Done() && counter < minibatch_size;
             feature_randomizer.Next(), label_randomizer.Next(),
             ref_label_randomizer.Next(), delta_randomizer.Next()){

          // get block of feature/delta pairs
          const vector<CuMatrix<BaseFloat>* >      &nnet_feat_in   = feature_randomizer.Value();
          const vector<vector< vector<uchar>* >* > &nnet_label_in  = label_randomizer.Value();
          const vector<vector<uchar>* >            &nnet_ref_label = ref_label_randomizer.Value();
          const vector<Vector<BaseFloat>* >        &delta          = delta_randomizer.Value();

          assert( nnet_feat_in.size() == 1 && nnet_label_in.size() == 1 &&
                nnet_ref_label.size() == 1 && delta.size() == 1);

          // Forward pass
          nnet.Feedforward(*nnet_feat_in[0], *nnet_label_in[0], &nnet_out);

          int maxIdx = strt.Eval(*delta[0], nnet_out, NULL, NULL, 0);

          if(maxIdx >= 0 && !crossvalidate){
             picked_nnet_feat_in[counter]   = nnet_feat_in[0];
             picked_nnet_label_in[counter]  = (*nnet_label_in[0])[maxIdx];
             picked_nnet_ref_label[counter] = nnet_ref_label[0];
             picked_delta[counter]          = (*delta[0])(maxIdx);
             counter++;
          }

          num_done += nnet_label_in[0]->size();
       }

       // backward pass
       if (!crossvalidate && counter != 0) {

          picked_nnet_feat_in.resize(counter);
          picked_nnet_label_in.resize(counter);
          picked_nnet_ref_label.resize(counter);
          picked_delta.resize(counter);

          Vector<BaseFloat> delta(counter);
          for(int i = 0; i < counter; ++i)
             delta(i) = picked_delta[i];

          nnet.Propagate(picked_nnet_feat_in, picked_nnet_label_in,
                picked_nnet_ref_label, &picked_nnet_out);

          strt.Eval(delta, picked_nnet_out, &obj_diff, NULL);
       
          // backpropagate
          nnet.Backpropagate(obj_diff, picked_nnet_label_in, picked_nnet_ref_label);
       }

    }

    // after last minibatch : show what happens in network 
    if (kaldi::g_kaldi_verbose_level >= 1 && !crossvalidate) { // vlog-1
       KALDI_VLOG(1) << "###############################";
       KALDI_VLOG(1) << nnet.InfoPropagate();
       KALDI_VLOG(1) << nnet.InfoBackPropagate();
       KALDI_VLOG(1) << nnet.InfoGradient();
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

