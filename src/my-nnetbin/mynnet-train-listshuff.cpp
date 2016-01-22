#include "nnet/nnet-trnopts.h"
#include "nnet/nnet-nnet.h"
#include "nnet/nnet-loss.h"
#include "nnet/nnet-randomizer.h"
#include "nnet/nnet-randomizer.cc"  // for template class
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "base/timer.h"
#include "cudamatrix/cu-device.h"

#include "my-nnet/nnet-my-nnet.h"
#include "score-path/score-path.h"
#include "my-nnet/nnet-my-loss.h"
#include <sstream>
#include <omp.h>

using namespace std;
using namespace kaldi;
using namespace kaldi::nnet1;

struct by_first { 
   bool operator()(pair<int, int> const &a, pair<int, int> const &b) const { 
      return a.first > b.first;
   }
};

int main(int argc, char *argv[]) {
  
  try {
    string usage;
    usage.append("Perform one iteration of My Neural Network training by shuffle mini-batch Stochastic Gradient Descent on training lists. Use learning to rank techniques. \n")
       .append("Use feature, label and path to train the neural net. \n")
       .append("Usage: ").append(argv[0]).append(" [options] <feature-rspecifier> <label-rspecifier> <score-path-rspecifier> <nnet-in> [<nnet-out>]\n")
       .append("e.g.: \n")
       .append(" ").append(argv[0]).append(" ark:feat.ark ark:lab.ark \"ark:lattice-to-vec ark:1.lat ark:- |\" nnet.init nnet.iter1\n");

    ParseOptions po(usage.c_str());

    NnetTrainOptions trn_opts;
    trn_opts.Register(&po);


    bool binary = true, 
         crossvalidate = false,
         randomize = true;

    po.Register("binary", &binary, "Write model in binary mode");
    po.Register("cross-validate", &crossvalidate, "Perform cross-validation (don't backpropagate)");

    NnetDataRandomizerOptions rnd_opts;
    rnd_opts.Register(&po);
    po.Register("randomize", &randomize, "Set training procedure randomized.");

    string acc_func = "pfac";
    po.Register("acc-func", &acc_func, "Acuracy function : fac|pac|pfac");

    string use_gpu="yes";
    po.Register("use-gpu", &use_gpu, "yes|no|optional, only has effect if compiled with CUDA");

    double dropout_retention = 0.0;
    po.Register("dropout-retention", &dropout_retention, "number between 0..1, saying how many neurons to preserve (0.0 will keep original value)");
     
    int negative_num = 0;
    po.Register("negative-num", &negative_num, "insert negative example in training");

    string feature_transform;
    po.Register("feature-transform", &feature_transform, "Feature transform in front of main network (in nnet format)");

    double error_margin = 0.02;
    po.Register("error-margin", &error_margin, "train on pairs with: |acc(x1) - acc(x2)| >= error_margin");
    double sigma = 1.0;
    po.Register("sigma", &sigma, "parameters of ranknet.");

    string loss_func = "listnet";
    po.Register("loss-func", &loss_func, "training loss function: (listnet, listrelu, ranknet)");

    // lstm parameters
    int32 num_stream = 8;
    po.Register("num-stream", &num_stream, "---LSTM--- BPTT multi-stream training");
    // lstm parameters

    po.Read(argc, argv);

    if (po.NumArgs() != 5-(crossvalidate?1:0)) {
      po.PrintUsage();
      exit(1);
    }

    // setup input parameters

    string feat_rspecifier       = po.GetArg(1),
           label_rspecifier      = po.GetArg(2),
           score_path_rspecifier = po.GetArg(3),
           nnet_in_filename      = po.GetArg(4),
           nnet_out_filename;

    if(!crossvalidate){
       nnet_out_filename = po.GetArg(5);
    }

    // function pointer used in calculating target.
    double (*acc_function)(const vector<uchar>& path1, const vector<uchar>& path2, bool norm);

    if(acc_func == "fac")
       acc_function = frame_acc;
    else if(acc_func == "pac")
       acc_function = phone_acc; 
    else if(acc_func == "pfac")
       acc_function = phone_frame_acc; 
    else{
       po.PrintUsage();
       exit(1);
    }

    StrtListBase* strt = StrtListBase::getInstance(loss_func, sigma, error_margin);
    
    if(strt == NULL)
       po.PrintUsage();

    //Select the GPU
#if HAVE_CUDA==1
    //sleep a while to get lock
    LockSleep(GPU_FILE);
    CuDevice::Instantiate().SelectGpuId(use_gpu);
#endif
    
    MyNnet nnet;
    nnet.Read(nnet_in_filename);
    nnet.SetTrainOptions(trn_opts);

    int labelNum = nnet.GetLabelNum();

    Nnet nnet_transf;
    if (feature_transform != "") {
       nnet_transf.Read(feature_transform);
    }

    if (dropout_retention > 0.0) {
      nnet.SetDropoutRetention(dropout_retention);
    }

    if (crossvalidate) {
      nnet.SetDropoutRetention(1.0);
    }

    Timer time;
    KALDI_LOG << (crossvalidate?"CROSS-VALIDATION":"TRAINING") << " STARTED";

    // ------------------------------------------------------------
    // read in all features and save to CPU (for shuffling)
    // read in all labels and save to CPU (for shuffling)
    SequentialScorePathReader         score_path_reader(score_path_rspecifier);
    SequentialBaseFloatMatrixReader   feature_reader(feat_rspecifier);
    SequentialUcharVectorReader       label_reader(label_rspecifier);

    vector< Matrix<BaseFloat> >       feature_arr;
    vector< vector< vector<uchar> > > label_arr;
    vector< vector<BaseFloat> >       target_arr;
    vector< int >                     length_arr;

    int max_length = -1;
    int num_Total = 0;
    int cols = -1; int seqs_stride = -1;

    srand(rnd_opts.randomizer_seed);

    for(; !(score_path_reader.Done() || feature_reader.Done() || label_reader.Done());
          score_path_reader.Next(), feature_reader.Next(), label_reader.Next()){

       assert( score_path_reader.Key() == feature_reader.Key() );
       assert( label_reader.Key() == feature_reader.Key() );

       const ScorePath::Table  &table = score_path_reader.Value().Value();
       const Matrix<BaseFloat> &feat  = feature_reader.Value();
       const vector<uchar>     &label = label_reader.Value();

       if(cols < 0) cols = feat.NumCols();
       assert( cols == feat.NumCols() );

       feature_arr.push_back(feat);
       length_arr.push_back(feat.NumRows());
       if(max_length < feat.NumRows()) max_length = feat.NumRows();

       target_arr.push_back(vector<BaseFloat>());
       label_arr.push_back(vector< vector<uchar> >());

       vector< vector<uchar> > &seqs = label_arr[label_arr.size() - 1];
       vector< BaseFloat >     &tgts = target_arr[target_arr.size() -1];

       seqs.reserve(1 + table.size() + negative_num);

       seqs.push_back(label);
       for(int i = 0; i < table.size(); ++i)
          seqs.push_back(table[i].second);

       vector<uchar> neg_arr(label.size());
       for(int i = 0; i < negative_num; ++i){
          for(int j = 0; j < neg_arr.size(); ++j)
             neg_arr[j] = rand() % labelNum + 1;
          seqs.push_back(neg_arr);
       }

       if(seqs_stride < 0) seqs_stride = seqs.size();
       assert(seqs_stride == seqs.size());

       tgts.resize(seqs.size());
#pragma omp parallel for
       for(int i = 0; i < seqs.size(); ++i)
          tgts[i] = acc_function(label, seqs[i], true);

       num_Total += seqs.size();
    }

    KALDI_LOG << "All Data Loaded.";
    
    vector<int> shuffle_idx(length_arr.size());
    for(int i = 0; i < length_arr.size(); ++i)
       shuffle_idx[i] = i;

    if(randomize && !crossvalidate){
       for(int i = 0; i < shuffle_idx.size(); ++i){
          int j = rand() % shuffle_idx.size();

          // swap
          int tmp = shuffle_idx[i];
          shuffle_idx[i] = shuffle_idx[j];
          shuffle_idx[j] = tmp;
       }
    }

    strt->SetAll(num_Total);

    CuMatrix<BaseFloat> nnet_in;
    MyCuMatrix<BaseFloat> nnet_out;
    CuMatrix<BaseFloat> nnet_out_diff;
    vector< CuMatrix<BaseFloat> > nnet_out_diff_arr(num_stream);
    vector< CuMatrix<BaseFloat> > features(num_stream); 

    int64 num_done = 0;
    int   now_idx  = 0;
    
    while(1){
       // filled in training array.
       int streams = 0;
       int max_T = 0;
       for(; streams < num_stream && now_idx < length_arr.size(); ++streams, ++now_idx){
          const Matrix<BaseFloat> & src = feature_arr[shuffle_idx[now_idx]];
          if(max_T < src.NumRows()) max_T = src.NumRows();
          
          nnet_transf.Feedforward(CuMatrix<BaseFloat>(src), &features[streams]);
       }
       if(streams == 0) break;

       nnet_in.Resize(max_T * streams, features[0].NumCols() , kSetZero);
       fillin(nnet_in, features, streams);

       // construct labels input
       vector<uchar> labels_in(max_T * streams * seqs_stride, 0);
       vector<int32> seq_length(streams, 0);

#pragma omp for
       for(int i = 0; i < streams; ++i){
          //label
          const vector< vector<uchar> > &label = label_arr[shuffle_idx[now_idx - streams + i]];
          for(int j = 0; j < label.size(); ++j)
             for(int k = 0; k < label[j].size(); ++k)
                labels_in[ seqs_stride * streams * k  + seqs_stride * i + j] =
                   label[j][k] - 1;

          seq_length[i] = label[0].size();
       }

       // setup nnet input
       nnet.SetLabelSeqs(labels_in, seqs_stride);

       vector<int32> flag(streams , 1);
       nnet.ResetLstmStreams(flag);
       nnet.SetSeqLengths(seq_length);

       // propagate nnet output
       nnet.Propagate(nnet_in, &nnet_out);

       assert(nnet_out.NumRows() == seqs_stride * streams);
       nnet_out_diff.Resize(nnet_out.NumRows(), nnet_out.NumCols(), kSetZero);

       for(int i = 0; i < streams; ++i){
          strt->Eval(target_arr[shuffle_idx[now_idx - streams + i]],
                nnet_out.RowRange(i*seqs_stride, seqs_stride), &nnet_out_diff_arr[i]);
          nnet_out_diff.RowRange(i*seqs_stride, seqs_stride).CopyFromMat(nnet_out_diff_arr[i]);
       }

       if(!crossvalidate){
          nnet.Backpropagate(nnet_out_diff, NULL);
       }

       num_done += streams * seqs_stride;
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
       nnet.Write(nnet_out_filename, binary);
    }

    KALDI_LOG << "Done " << num_done << " examples, " 
              << " with other errors. "
              << "[" << (crossvalidate?"CROSS-VALIDATION":"TRAINING")
              << ", " << time.Elapsed()/60 << " min"
              << "]";  

    KALDI_LOG << strt->Report();

#if HAVE_CUDA==1
    CuDevice::Instantiate().PrintProfile();
#endif
    cudaDeviceReset();

    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

