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
#include "nnet-my-nnet.h"
#include <sstream>
#include <omp.h>

using namespace std;
using namespace kaldi;
using namespace kaldi::nnet1;

void fillin(CuMatrixBase<BaseFloat> &dest, vector< CuMatrix<BaseFloat> > &src, int stream_num);
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

    string acc_func = "fac";
    po.Register("acc-func", &acc_func, "Acuracy function : fac|pac");

    string use_gpu="yes";
    po.Register("use-gpu", &use_gpu, "yes|no|optional, only has effect if compiled with CUDA");

    double dropout_retention = 0.0;
    po.Register("dropout-retention", &dropout_retention, "number between 0..1, saying how many neurons to preserve (0.0 will keep original value)");
     
    int negative_num = 0;
    po.Register("negative-num", &negative_num, "insert negative example in training");

    string feature_transform;
    po.Register("feature-transform", &feature_transform, "Feature transform in front of main network (in nnet format)");

    double error_margin = 0.05;
    po.Register("error-margin", &error_margin, "train on pairs with: |acc(x1) - acc(x2)| >= error_margin");
    double sigma = 1.0;
    po.Register("sigma", &sigma, "parameters of ranknet.");

    string loss_func = "listnet";
    po.Register("loss-func", &loss_func, "training loss function: (listnet, listrelu, ranknet)");

    // lstm parameters
    int32 num_stream = 2;
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
    vector< pair< int, int >  >       length_arr;

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
       length_arr.push_back(make_pair(feat.NumRows(), length_arr.size()));

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

    // random shuffle
    for(int i = 0; i < length_arr.size(); ++i){
       int j = rand() % length_arr.size();
       pair<int, int> tmp = length_arr[i];
       length_arr[i] = length_arr[j];
       length_arr[j] = tmp;
    }

    // sorting according to the length
    sort(length_arr.begin(), length_arr.end(), by_first());

    // constructing mini-batch
    vector< vector<int> > mini_batch_idx;
    for(int i = 0; i < length_arr.size(); ){
       vector<int> batch;
       for(int j = 0; j < num_stream && i < length_arr.size(); ++i, ++j){
          batch.push_back(length_arr[i].second);
       }
       mini_batch_idx.push_back(batch);
    }

    // random shuffle
    vector<int> shuffle_arr(mini_batch_idx.size());
    for(int i = 0; i < shuffle_arr.size(); ++i)
       shuffle_arr[i] = i;

    if(randomize && !crossvalidate){
       for(int i = 0; i < shuffle_arr.size(); ++i){
          int j = rand() % shuffle_arr.size();

          // swap
          int tmp = shuffle_arr[i];
          shuffle_arr[i] = shuffle_arr[j];
          shuffle_arr[j] = tmp;
       }
    }

    strt->SetAll(num_Total);

    CuMatrix<BaseFloat> nnet_out;
    CuMatrix<BaseFloat> nnet_out_diff;
    vector< CuMatrix<BaseFloat> > nnet_out_diff_arr(num_stream);

    int64 num_done = 0;
    
    for(int i = 0; i < shuffle_arr.size(); ++i){
       // filled in training array.
       vector< CuMatrix<BaseFloat> >     features(num_stream); // all features

       int maxT = 0; 

       const vector<int> & mini_batch = mini_batch_idx[shuffle_arr[i]];

       for(int j = 0; j < mini_batch.size(); ++j){
          const Matrix<BaseFloat> & feat = feature_arr[mini_batch[j]];
          if(maxT < feat.NumRows()) maxT = feat.NumRows();
          nnet_transf.Feedforward(CuMatrix<BaseFloat>(feat), &features[j]);
       }

       // construct labels input

       CuMatrix<BaseFloat> nnet_in(maxT * num_stream, cols, kSetZero);
       vector<int32>     labels_in(maxT * num_stream * seqs_stride, 0);

       fillin(nnet_in, features, num_stream);
#pragma omp parallel for
       for(int l = 0; l < mini_batch.size(); ++l){
          const vector< vector<uchar> > &label = label_arr[mini_batch[l]];
          for(int j = 0; j < label.size(); ++j)
             for(int k = 0; k < label[j].size(); ++k)
                labels_in[ seqs_stride * num_stream * k  + seqs_stride * l + j] =
                   label[j][k] - 1;
       }

       // setup nnet input
       nnet.SetLabelSeqs(labels_in, seqs_stride);

       vector<int32> flag(num_stream , 1);
       nnet.ResetLstmStreams(flag);

       vector<int32> seq_length(num_stream, 0);
       for(int l = 0; l < mini_batch.size(); ++l)
          seq_length[l] = label_arr[mini_batch[l]][0].size();
       nnet.SetSeqLengths(seq_length);

       // propagate nnet output
       nnet.Propagate(nnet_in, &nnet_out);

       assert(nnet_out.NumRows() == seqs_stride * num_stream);
       nnet_out_diff.Resize(nnet_out.NumRows(), nnet_out.NumCols(), kSetZero);
       for(int l = 0; l < mini_batch.size(); ++l){
          strt->Eval(target_arr[mini_batch[l]],
                nnet_out.RowRange(l*seqs_stride, seqs_stride), &nnet_out_diff_arr[l]);
          nnet_out_diff.RowRange(l * seqs_stride, seqs_stride).CopyFromMat(nnet_out_diff_arr[l]);
       }

       if(!crossvalidate){
          nnet.Backpropagate(nnet_out_diff, NULL);
       }

       num_done += mini_batch.size() * seqs_stride;
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

void fillin(CuMatrixBase<BaseFloat> &dest, vector< CuMatrix<BaseFloat> > &src, int stream_num){

   for(int i = 0; i < stream_num; ++i){
      BaseFloat *src_data = getCuPointer(&src[i]);
      BaseFloat *dest_data = getCuPointer(&dest) + dest.Stride() * i;
      size_t dst_pitch = dest.Stride() * sizeof(BaseFloat) * stream_num;
      size_t src_pitch = src[i].Stride() * sizeof(BaseFloat);
      size_t width     = src[i].NumCols() * sizeof(BaseFloat);
      size_t height    = src[i].NumRows();

      if(height != 0)
         CU_SAFE_CALL(cudaMemcpy2D(dest_data, dst_pitch, src_data, src_pitch,
                  width, height, cudaMemcpyDeviceToDevice));
   }
}
