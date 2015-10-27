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
#include "srnnet2.h"
#include <sstream>
#include <omp.h>

using namespace std;
using namespace kaldi;
using namespace kaldi::nnet1;

typedef struct{
   int labelID;
   int frameID;
} Frame;

typedef StdVectorRandomizer<Frame> FrameRandomizer;

int main(int argc, char *argv[]) {
  
  try {
    string usage;
    usage.append("Perform one epoch of Structure Recurrent Neural Network training by shuffle mini-batch Stochastic Gradient Descent on training frames. Use learning to rank techniques. \n")
       .append("Use feature, label and path to train the neural net. \n")
       .append("Usage: ").append(argv[0]).append(" [options] <feature-rspecifier> <label-rspecifier> <nnet-in> [<nnet-out>]\n")
       .append("e.g.: \n")
       .append(" ").append(argv[0]).append(" ark:feat.ark ark:lab.ark nnet.init nnet.iter1\n");

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

    string use_gpu="yes";
    po.Register("use-gpu", &use_gpu, "yes|no|optional, only has effect if compiled with CUDA");

    double dropout_retention = 0.0;
    po.Register("dropout-retention", &dropout_retention, "number between 0..1, saying how many neurons to preserve (0.0 will keep original value)");
     
    double nnet_ratio = 1.0;
    po.Register("nnet-ratio", &nnet_ratio, "nnet1 learning rate ratio");

    string feature_transform;
    po.Register("feature-transform", &feature_transform, "Feature transform in front of main network (in nnet format)");

    double sigma = 1.0;
    po.Register("sigma", &sigma, "parameters of ranknet.");

    int num_frames = -1;
    po.Register("num-frame", &num_frames, "limit the number of frames used in training(-1 for all training set).");

    int depth = 20;
    po.Register("depth", &depth, "BPTT depth");

    po.Read(argc, argv);

    if (po.NumArgs() != 4-(crossvalidate?1:0)) {
      po.PrintUsage();
      exit(1);
    }

    // setup input parameters

    string feat_rspecifier       = po.GetArg(1),
           label_rspecifier      = po.GetArg(2),
           nnet_in_filename      = po.GetArg(3),
           nnet_out_filename;

    if(!crossvalidate){
       nnet_out_filename = po.GetArg(4);
    }

    StrtBest* strt = new StrtBest(sigma);
    
    if(strt == NULL)
       po.PrintUsage();

    //Select the GPU
#if HAVE_CUDA==1
    //sleep a while to get lock
    LockSleep(GPU_FILE);
    CuDevice::Instantiate().SelectGpuId(use_gpu);
#endif
    SRNnet2 nnet;
    nnet.Read(nnet_in_filename);
    nnet.SetTrainOptions(trn_opts, nnet_ratio);

    //int stateMax = nnet.StateMax();

    // ------------------------------------------------------------
    // read in all features and save to GPU
    // read in all labels and save to CPU
    SequentialBaseFloatMatrixReader  feature_reader(feat_rspecifier);
    SequentialUcharVectorReader      label_reader(label_rspecifier);

    vector< CuMatrix<BaseFloat> > features; // all features
    vector< vector<uchar> >       ref_label; // positive examples

    features.reserve(BUFSIZE);
    ref_label.reserve(BUFSIZE);


    for ( ; !(feature_reader.Done() || label_reader.Done());
          feature_reader.Next(), label_reader.Next()) {

       assert( label_reader.Key() == feature_reader.Key() );

       const Matrix<BaseFloat> &feat  = feature_reader.Value();
       const vector<uchar>     &label = label_reader.Value();

       features.push_back(CuMatrix<BaseFloat>(feat));
       ref_label.push_back(label);
    } 
    // -------------------------------------------------------------
    //
    // fixed minibatch_size = 1
    rnd_opts.minibatch_size = 1;

    RandomizerMask        randomizer_mask(rnd_opts);
    FrameRandomizer       frame_randomizer(rnd_opts);

    {
       vector<Frame> frames;
       Frame         tmp_frame;
       for(int i = 0; i < ref_label.size(); ++i)
          for(int j = 0; j < ref_label[i].size(); ++j){
             tmp_frame.labelID = i;
             tmp_frame.frameID = j;
             frames.push_back(tmp_frame);
          }
       frame_randomizer.AddData(frames);
    }

    KALDI_ASSERT (!frame_randomizer.IsFull());
    
    KALDI_LOG << "Filled all randomizer. features # = " << features.size()
       << ", frame # = " << frame_randomizer.NumFrames();
    
    // randomize
    //if (!crossvalidate && randomize) {
    if (randomize) {
       const std::vector<int32>& mask = 
          randomizer_mask.Generate(frame_randomizer.NumFrames());
       frame_randomizer.Randomize(mask);
    }

    // prepare Nnet
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

    int total_frames = frame_randomizer.NumFrames();
    if(num_frames < 0 || num_frames > total_frames)
       num_frames = frame_randomizer.NumFrames();

    strt->SetAll(num_frames);

    for (; (!frame_randomizer.Done()) && num_done < num_frames; frame_randomizer.Next()){

#if HAVE_CUDA==1
       // check the GPU is not overheated
       CuDevice::Instantiate().CheckGpuHealth();
#endif
       assert(frame_randomizer.Value().size() == 1);
       const Frame &frm = frame_randomizer.Value()[0];

       const CuMatrix<BaseFloat> &nnet_feat_in = features[frm.labelID]; 
       const vector<uchar>       &ref_label_in = ref_label[frm.labelID];

       nnet.Propagate(nnet_feat_in, ref_label_in, frm.frameID, &nnet_out);

       strt->Eval(ref_label_in[frm.frameID] - 1, nnet_out, &obj_diff);

       // backpropagate
       if (!crossvalidate) {
          nnet.Backpropagate(obj_diff, ref_label_in, frm.frameID, depth);
       }

       num_done ++;
    }
    KALDI_LOG << "Done " << num_done << " frames.";


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
              << ", " << (randomize?"RANDOMIZED":"NOT-RANDOMIZED") 
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

