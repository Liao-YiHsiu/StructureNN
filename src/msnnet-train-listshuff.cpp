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
#include "msnnet.h"
#include <sstream>
#include <omp.h>

using namespace std;
using namespace kaldi;
using namespace kaldi::nnet1;

typedef StdVectorRandomizer<CuMatrix<BaseFloat>* >    MatrixPtRandomizer;
typedef StdVectorRandomizer<vector<vector<uchar> >* > LabelArrPtRandomizer;
typedef StdVectorRandomizer<vector<BaseFloat>* >      TargetArrPtRandomizer;

void fillin(CuMatrixBase<BaseFloat> &dest, vector< CuMatrix<BaseFloat> > &src, int stream_num);

int main(int argc, char *argv[]) {
  
  try {
    string usage;
    usage.append("Perform one iteration of Mux Structure Neural Network training by shuffle mini-batch Stochastic Gradient Descent on training lists. Use learning to rank techniques. \n")
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

    // <dummy> not used
    NnetDataRandomizerOptions rnd_opts;
    rnd_opts.Register(&po);
    po.Register("randomize", &randomize, "dummy option");
    // </dummy>

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
    
    MSNnet nnet;
    nnet.Read(nnet_in_filename);
    nnet.SetTrainOptions(trn_opts);

    int stateMax = nnet.StateMax();

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
    // read in all features and save to GPU
    // read in all labels and save to CPU
    SequentialScorePathReader        score_path_reader(score_path_rspecifier);
    SequentialBaseFloatMatrixReader  feature_reader(feat_rspecifier);
    SequentialUcharVectorReader      label_reader(label_rspecifier);

    int64 num_done = 0;
    int num_Total = 0;

    strt->SetAll(num_Total);
    srand(rnd_opts.randomizer_seed);

    CuMatrix<BaseFloat> nnet_out;
    CuMatrix<BaseFloat> nnet_out_diff;
    vector< CuMatrix<BaseFloat> > nnet_out_diff_arr(num_stream);
    
    while(1) {
       // filled in training array.
       vector< CuMatrix<BaseFloat> >     features(num_stream); // all features
       vector< vector<BaseFloat> >       targets(num_stream);
       vector< vector< vector<uchar> > > examples(num_stream);

       int snum;
       int maxT = 0; int cols = -1; int seqs_stride = -1;

       for(snum = 0; snum < num_stream; ++snum){
          if(score_path_reader.Done() || feature_reader.Done() || label_reader.Done())
             break;

          assert( score_path_reader.Key() == feature_reader.Key() );
          assert( label_reader.Key() == feature_reader.Key() );

          const ScorePath::Table  &table = score_path_reader.Value().Value();
          const Matrix<BaseFloat> &feat  = feature_reader.Value();
          const vector<uchar>     &label = label_reader.Value();

          if(maxT < feat.NumRows()) maxT = feat.NumRows();
          
          
          nnet_transf.Feedforward(CuMatrix<BaseFloat>(feat), &features[snum]);
          if(cols < 0) cols = features[snum].NumCols();
          assert( cols == features[snum].NumCols() );

          vector< vector<uchar> > &seqs = examples[snum];
          vector< BaseFloat >     &tgts = targets[snum];

          seqs.reserve(1 + table.size() + negative_num);

          seqs.push_back(label);
          for(int i = 0; i < table.size(); ++i)
             seqs.push_back(table[i].second);

          vector<uchar> neg_arr(label.size());
          for(int i = 0; i < negative_num; ++i){
             for(int j = 0; j < neg_arr.size(); ++j)
                neg_arr[j] = rand() % stateMax + 1;
             seqs.push_back(neg_arr);
          }

          if(seqs_stride < 0) seqs_stride = seqs.size();
          assert(seqs_stride == seqs.size());

          tgts.resize(seqs.size());
#pragma omp parallel for
          for(int i = 0; i < seqs.size(); ++i)
             tgts[i] = acc_function(label, seqs[i], true);

          num_Total += seqs.size();

          score_path_reader.Next();
          feature_reader.Next();
          label_reader.Next();
       }

       if(snum == 0)break;

       // construct labels input

       CuMatrix<BaseFloat> nnet_in(maxT * snum, cols, kSetZero);
       vector<int32>     labels_in(maxT * snum * seqs_stride, 0);

       fillin(nnet_in, features, snum);
#pragma omp parallel for
       for(int i = 0; i < snum; ++i)
          for(int j = 0; j < seqs_stride; ++j)
             for(int k = 0; k < examples[i][j].size(); ++k)
                labels_in[ seqs_stride * snum * k  + seqs_stride * i + j] = examples[i][j][k] - 1;


       // setup nnet input
       nnet.SetLabelSeqs(labels_in, seqs_stride);

       vector<int32> flag(snum, 1);
       nnet.ResetLstmStreams(flag);

       vector<int32> seq_length(snum);
       for(int i = 0; i < snum; ++i)
          seq_length[i] = examples[i][0].size();
       nnet.SetSeqLengths(seq_length);

       // propagate nnet output
       nnet.Propagate(nnet_in, &nnet_out);

       nnet_out_diff.Resize(seqs_stride*snum, nnet_out.NumCols(), kUndefined);
       for(int i = 0; i < snum; ++i){
          strt->Eval(targets[i], nnet_out.RowRange(i*seqs_stride, seqs_stride), &nnet_out_diff_arr[i]);
          nnet_out_diff.RowRange(i * seqs_stride, seqs_stride).CopyFromMat(nnet_out_diff_arr[i]);
       }

       if(!crossvalidate){
          nnet.Backpropagate(nnet_out_diff, NULL);
       }

       num_done += snum * seqs_stride;
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

      CU_SAFE_CALL(cudaMemcpy2D(dest_data, dst_pitch, src_data, src_pitch,
               width, height, cudaMemcpyDeviceToDevice));
   }
}
