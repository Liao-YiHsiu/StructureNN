#include "nnet/nnet-trnopts.h"
#include "nnet/nnet-nnet.h"
#include "nnet/nnet-loss.h"
#include "nnet/nnet-randomizer.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "base/timer.h"
#include "cudamatrix/cu-device.h"
#include "svm.h"
#include <sstream>

using namespace std;
using namespace kaldi;
using namespace kaldi::nnet1;



int main(int argc, char *argv[]) {
  
  try {
    string usage;
    usage.append("Perform one iteration of Structure Neural Network training by mini-batch Stochastic Gradient Descent.\n")
       .append("Use feature, label and path to train the neural net. \n")
       .append("Usage: ").append(argv[0]).append(" [options] <feature-rspecifier> <label-rspecifier> <score-path-rspecifier> <model-in> [<model-out>]\n")
       .append("e.g.: \n")
       .append(" ").append(argv[0]).append(" ark:feat.ark ark:lab.ark \"ark:lattice-to-vec ark:1.lat ark:- |\" nnet.init nnet.iter1\n");

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

    double power = 1.0;
    po.Register("power", &power, "transform the scoring function");

    double insertion_penalty = 1.0;
    po.Register("insertion_penalty", &insertion_penalty, "PER calculate insertion penalty");

    string use_gpu="yes";
    po.Register("use-gpu", &use_gpu, "yes|no|optional, only has effect if compiled with CUDA");

    double dropout_retention = 0.0;
    po.Register("dropout-retention", &dropout_retention, "number between 0..1, saying how many neurons to preserve (0.0 will keep original value)");
     
    int max_state = 48;
    po.Register("max-state", &max_state, "max state ID");

    int negative_num = 0;
    po.Register("negative-num", &negative_num, "insert negative example in training");

    bool reweight = false;
    po.Register("reweight", &reweight, "reweight training examles");

    
    po.Read(argc, argv);

    if (po.NumArgs() != 5-(crossvalidate?1:0)) {
      po.PrintUsage();
      exit(1);
    }


    string feat_rspecifier  = po.GetArg(1),
      label_rspecifier      = po.GetArg(2),
      score_path_rspecifier = po.GetArg(3),
      model_filename        = po.GetArg(4),
      target_model_filename;

    // function pointer used in calculating target.
    double (*acc_function)(const vector<int32>& path1, const vector<int32>& path2, double param);

    if(error_function == "fer")
       acc_function = frame_acc;
    else if(error_function == "per")
       acc_function = phone_acc; 
    else{
       po.PrintUsage();
       exit(1);
    }

        
    if (!crossvalidate) {
      target_model_filename = po.GetArg(5);
    }

    RandomizerMask        randomizer_mask(rnd_opts);
    MatrixRandomizer      feature_randomizer(rnd_opts);
    PosteriorRandomizer   targets_randomizer(rnd_opts);
    VectorRandomizer      weights_randomizer(rnd_opts);

    SequentialScorePathReader        score_path_reader(score_path_rspecifier);
    SequentialBaseFloatMatrixReader  feature_reader(feat_rspecifier);
    SequentialInt32VectorReader      label_reader(label_rspecifier);

    //Select the GPU
#if HAVE_CUDA==1
    CuDevice::Instantiate().SelectGpuId(use_gpu);
    CuDevice::Instantiate().DisableCaching();
#endif

    Nnet nnet_transf;

    Nnet nnet;
    nnet.Read(model_filename);
    nnet.SetTrainOptions(trn_opts);

    if (dropout_retention > 0.0) {
      nnet.SetDropoutRetention(dropout_retention);
    }

    if (crossvalidate) {
      nnet.SetDropoutRetention(1.0);
    }

    kaldi::int64 total_frames = 0;

    Xent xent;
    Mse mse;
    
    CuMatrix<BaseFloat> nnet_out, obj_diff;

    Timer time;
    KALDI_LOG << (crossvalidate?"CROSS-VALIDATION":"TRAINING") << " STARTED";

    int32 featsN = -1;
    int32 num_done = 0, num_no_tgt_mat = 0, num_other_error = 0;
    while(!score_path_reader.Done()){
#if HAVE_CUDA==1
      // check the GPU is not overheated
      CuDevice::Instantiate().CheckGpuHealth();
#endif
      // fill the randomizer
      for ( ; !score_path_reader.Done();
            score_path_reader.Next(), feature_reader.Next(), label_reader.Next()) {

        if (feature_randomizer.IsFull()) break; // suspend, keep utt for next loop

        assert( score_path_reader.Key() == feature_reader.Key() );
        assert( label_reader.Key() == feature_reader.Key() );

        const ScorePath::Table  &table = score_path_reader.Value().Value();
        const Matrix<BaseFloat> &feat  = feature_reader.Value();
        const vector<int32>     &label = label_reader.Value();

        if(featsN < 0)
           featsN = (feat.NumCols() + max_state) * max_state;
        assert((feat.NumCols() + max_state)*max_state == featsN);

        Matrix<BaseFloat> feats(table.size() + 1 + negative_num, featsN);
        Posterior         targets;

        // positive example
        makeFeature(feat, label, max_state, feats.Row(0));

        makePost(pow(acc_function(label, label, insertion_penalty), power), targets);

        // input example
        for(int i = 0; i < table.size(); ++i){
           makeFeature(feat, table[i].second, max_state, feats.Row(i+1));

           makePost(pow(acc_function(label, table[i].second, insertion_penalty), power), targets);
        }

        // random negitive example
        vector<int32> neg_arr(label.size());
        for(int i = 0; i < negative_num; ++i){
           for(int j = 0; j < neg_arr.size(); ++j)
              neg_arr[j] = rand() % max_state + 1;
           makeFeature(feat, neg_arr, max_state, feats.Row(table.size()+1+i));
           makePost(pow(acc_function(label, neg_arr, insertion_penalty), power), targets);
        }

        // TODO: pos example weight can be larger.
        Vector<BaseFloat> weights;
        weights.Resize(feats.NumRows());
        weights.Set(1.0);

        // reweight input example
        if(reweight){
           double decay = 1;
           for(int i = 0; i < table.size(); ++i){
              weights(i+1) *= decay;
              decay *= 0.9;
           }
           weights.Range(1 + table.size(), negative_num).Scale(1/(double)negative_num);
        }

        
        feature_randomizer.AddData(CuMatrix<BaseFloat>(feats));
        targets_randomizer.AddData(targets);
        weights_randomizer.AddData(weights);
        num_done++;
      
        // report the speed
        if (num_done % 5000 == 0) {
          double time_now = time.Elapsed();
          KALDI_VLOG(1) << "After " << num_done << " utterances: time elapsed = "
                        << time_now/60 << " min; processed " << total_frames/time_now
                        << " frames per second.";
        }
      }

      // randomize
      if (!crossvalidate && randomize) {
        const std::vector<int32>& mask = randomizer_mask.Generate(feature_randomizer.NumFrames());
        feature_randomizer.Randomize(mask);
        targets_randomizer.Randomize(mask);
        weights_randomizer.Randomize(mask);
      }

      // train with data from randomizers (using mini-batches)
      for ( ; !feature_randomizer.Done(); feature_randomizer.Next(),
                                          targets_randomizer.Next(),
                                          weights_randomizer.Next()){
        // get block of feature/target pairs
        const CuMatrixBase<BaseFloat>& nnet_in = feature_randomizer.Value();
        const Posterior& nnet_tgt = targets_randomizer.Value();
        const Vector<BaseFloat>& frm_weights = weights_randomizer.Value();

        // forward pass
        nnet.Propagate(nnet_in, &nnet_out);

        // evaluate objective function we've chosen
        if (objective_function == "xent") {
          //xent.Eval(nnet_out, nnet_tgt, &obj_diff);
          xent.Eval(frm_weights, nnet_out, nnet_tgt, &obj_diff); 
        } else if (objective_function == "mse") {
          //mse.Eval(nnet_out, nnet_tgt, &obj_diff);
          mse.Eval(frm_weights, nnet_out, nnet_tgt, &obj_diff);
        } else {
          KALDI_ERR << "Unknown objective function code : " << objective_function;
        }

        // backward pass
        if (!crossvalidate) {
          // backpropagate
          nnet.Backpropagate(obj_diff, NULL);
        }

        // 1st minibatch : show what happens in network 
        if (kaldi::g_kaldi_verbose_level >= 1 && total_frames == 0) { // vlog-1
          KALDI_VLOG(1) << "### After " << total_frames << " frames,";
          KALDI_VLOG(1) << nnet.InfoPropagate();
          if (!crossvalidate) {
            KALDI_VLOG(1) << nnet.InfoBackPropagate();
            KALDI_VLOG(1) << nnet.InfoGradient();
          }
        }
        
        // monitor the NN training
        if (kaldi::g_kaldi_verbose_level >= 2) { // vlog-2
          if ((total_frames/25000) != ((total_frames+nnet_in.NumRows())/25000)) { // print every 25k frames
            KALDI_VLOG(2) << "### After " << total_frames << " frames,";
            KALDI_VLOG(2) << nnet.InfoPropagate();
            if (!crossvalidate) {
              KALDI_VLOG(2) << nnet.InfoGradient();
            }
          }
        }
        
        total_frames += nnet_in.NumRows();
      }
    }
    
    // after last minibatch : show what happens in network 
    if (kaldi::g_kaldi_verbose_level >= 1) { // vlog-1
      KALDI_VLOG(1) << "### After " << total_frames << " frames,";
      KALDI_VLOG(1) << nnet.InfoPropagate();
      if (!crossvalidate) {
        KALDI_VLOG(1) << nnet.InfoBackPropagate();
        KALDI_VLOG(1) << nnet.InfoGradient();
      }
    }

    if (!crossvalidate) {
      nnet.Write(target_model_filename, binary);
    }

    KALDI_LOG << "Done " << num_done << " files, " << num_no_tgt_mat
              << " with no tgt_mats, " << num_other_error
              << " with other errors. "
              << "[" << (crossvalidate?"CROSS-VALIDATION":"TRAINING")
              << ", " << (randomize?"RANDOMIZED":"NOT-RANDOMIZED") 
              << ", " << time.Elapsed()/60 << " min, fps" << total_frames/time.Elapsed()
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

