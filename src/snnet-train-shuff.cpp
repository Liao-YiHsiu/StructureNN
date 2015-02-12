#include "nnet/nnet-trnopts.h"
#include "nnet/nnet-nnet.h"
#include "nnet/nnet-loss.h"
#include "nnet/nnet-randomizer.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "base/timer.h"
#include "cudamatrix/cu-device.h"
#include <sstream>

using namespace std;
using namespace kaldi;
using namespace kaldi::nnet1;

typedef pair< vector<int32>, vector< vector<float> > > LabFeat;
typedef vector< LabFeat > LabFeatVec;

void svm_read(const string &file, LabFeatVec &labfeat); 
int path_read(const string &path_rspecifier, const LabFeatVec& labfeats,
      vector< vector< vector<int32> > > &all_path);
void permute_arr(vector<int> &arr);

void makeFeature(Matrix<BaseFloat> &matrix, const LabFeat& labfeat, const vector<int32> &path, int32 maxState);
void makePost(Posterior &post, const vector<int32> &realPath, const vector<int32> &path);

void train(const LabFeatVec &labfeats, const vector< vector< vector<int32> > > &all_path,
      Nnet &nnet, const vector<int> &randIndex, int pathN,
      const NnetDataRandomizerOptions& rnd_opts, bool crossvalidate,
      const string &target_model_filename, bool binary, bool randomize,
      const string &objective_function, int maxState);

int main(int argc, char *argv[]) {
  
  try {
    string usage =
        "Perform one iteration of Structure Neural Network training by mini-batch Stochastic Gradient Descent.\n"
        "Use structure SVM file and path to train the neural net. \n"
        "Train by SGD and cut Cross Validation set automatically.\n"
        "Usage:  snnet-train-shuff [options] <SVM in> <path-rspecifier> <model-in> <model-out>\n"
        "e.g.: \n"
        " nnet-train-frmshuff data.out \"ark:lattice-to-vec ark:1.lat ark:- |\" nnet.init nnet.iter1\n";

    ParseOptions po(usage.c_str());

    NnetTrainOptions trn_opts;
    trn_opts.Register(&po);
    NnetDataRandomizerOptions rnd_opts;
    rnd_opts.Register(&po);


    bool binary = true, 
         randomize = true;

    po.Register("binary", &binary, "Write output in binary mode");
    po.Register("randomize", &randomize, "Perform the frame-level shuffling within the Cache::");

    string objective_function = "xent";
    po.Register("objective-function", &objective_function, "Objective function : xent|mse");

 //   int32 length_tolerance = 5;
 //   po.Register("length-tolerance", &length_tolerance, "Allowed length difference of features/targets (frames)");
    
    string use_gpu="yes";
    po.Register("use-gpu", &use_gpu, "yes|no|optional, only has effect if compiled with CUDA");

    double cv_percent = 10;
    po.Register("cv-percent", &cv_percent, "Cross Validation percentage.(default = 10)");
    
    double dropout_retention = 0.0;
    po.Register("dropout-retention", &dropout_retention, "number between 0..1, saying how many neurons to preserve (0.0 will keep original value");
     
    
    po.Read(argc, argv);

    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }

    string svm_file = po.GetArg(1),
      path_rspecifier = po.GetArg(2),
      model_filename = po.GetArg(3),
      target_model_filename = po.GetArg(4);
        
    //Select the GPU
#if HAVE_CUDA==1
    CuDevice::Instantiate().SelectGpuId(use_gpu);
    CuDevice::Instantiate().DisableCaching();
#endif


    Nnet nnet;
    nnet.Read(model_filename);
    nnet.SetTrainOptions(trn_opts);

    if (dropout_retention > 0.0) {
      nnet.SetDropoutRetention(dropout_retention);
    }

    LabFeatVec labfeats; 
    svm_read(svm_file, labfeats);


    vector< vector< vector<int32> > > all_path;
    int maxState = path_read(path_rspecifier, labfeats, all_path);

    // cut data into Cross Validation set and Training set
    int pathN = all_path[0].size();

    assert(all_path.size() * labfeats.size());
    int N = labfeats.size()*pathN;
    int cvN = N * cv_percent / 100;
    int trN = N - cvN;

    srand(rnd_opts.randomizer_seed);

    vector<int> randIndex(trN);
    permute_arr(randIndex);

    train(labfeats, all_path, nnet, randIndex, pathN, 
          rnd_opts, false, target_model_filename, binary, 
          randomize, objective_function, maxState);

    randIndex.resize(cvN);
    permute_arr(randIndex);
    for(int i = 0; i < randIndex.size(); ++i)
       randIndex[i] += trN;

    train(labfeats, all_path, nnet, randIndex, pathN, 
          rnd_opts, true, target_model_filename, binary, 
          randomize, objective_function, maxState);

#if HAVE_CUDA==1
    CuDevice::Instantiate().PrintProfile();
#endif

    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

void svm_read(const string &file, LabFeatVec &labfeats){
   ifstream fin(file.c_str());
   string line;

   string uid, puid;
   int label;
   int id;
   float value;
   char c;

   int feat_dim = 0;

   vector<int32> labels;
   vector< vector<float> > feats;

   while(getline(fin, line)){
      stringstream ss(line);
      ss >> label >> uid;

      vector<float> arr;
      if(feat_dim != 0)
         arr.resize(feat_dim, 0);

      while(ss >> id >> c >> value){
         if(feat_dim == 0){
            if(arr.size() < id)
               arr.resize(id, 0);
         }else
            assert(feat_dim >= id);

         arr[id-1] = value;
      }

      feat_dim = feat_dim == 0 ? arr.size(): feat_dim;


      if( uid != puid && puid != "" ){
         labfeats.push_back(make_pair(labels, feats));
         labels.clear(); feats.clear();
      }

      puid = uid;
      labels.push_back(label);
      feats.push_back(arr);
   }

   labfeats.push_back(make_pair(labels, feats));
}

int path_read(const string &path_rspecifier, const LabFeatVec& labfeats,
      vector< vector< vector<int32> > > &all_path){

    int32 maxState = -1;
    SequentialInt32VectorVectorReader path_reader(path_rspecifier);

    for(int i = 0; !path_reader.Done(); ++i, path_reader.Next()){
       all_path.push_back(path_reader.Value());
       vector< vector<int32> > &path = all_path[all_path.size() - 1];
       assert(path[0].size() == labfeats[i].first.size());

       // put correct answer into training data.
       path.push_back(labfeats[i].first);

       for(int j = 0; j < path.size(); ++j){
          for(int k = 0; k < path[j].size(); ++k)
             maxState = maxState < path[j][k] ? path[j][k] : maxState;
       }
       
    }
    return maxState;
}

void permute_arr(vector<int> &arr){
   int size = arr.size();

   for(int i = 0; i < size; ++i)
      arr[i] = i;

   for(int i = 0; i < size; ++i){
      int j = rand() % size;
      int tmp = arr[j];
      arr[j] = arr[i];
      arr[i] = tmp;
   }
}


// TODO: feature normalization
void makeFeature(Matrix<BaseFloat> &matrix, const LabFeat& labfeat, const vector<int32> &path, int32 maxState){
   const vector< vector<float> >& feat = labfeat.second;
   int feat_dim = feat[0].size();

   int N = feat_dim * maxState + maxState * maxState;
   matrix.Resize(1, N);

   SubVector<BaseFloat> vec(matrix, 0);

   SubVector<BaseFloat> tran(vec, feat_dim * maxState, maxState*maxState);
   for(int i = 0; i < path.size(); ++i){
      SubVector<BaseFloat> obs(vec, path[i]*feat_dim, feat_dim);
      for(int j = 0; j < feat_dim; ++j)
         obs(j) += feat[i][j];
      if(i > 0){
         tran(path[i-1]*maxState + path[i]) += 1;
      }
   }

   // normalization
   for(int i = 0; i < N; ++i)
      vec(i) /= path.size();

}

void makePost(Posterior &post, const vector<int32> &realPath, const vector<int32> &path){
   assert(realPath.size() == path.size());
   int corr = 0;
   for(int i = 0; i < path.size(); ++i){
      corr += realPath[i] == path[i] ? 1:0;
   }

   double err = corr / (double)path.size();
   vector< pair<int32, BaseFloat> > arr; 

   arr.push_back(make_pair(1, err));
   arr.push_back(make_pair(1, 1-err));
   post.push_back(arr);

}

void train(const LabFeatVec &labfeats, const vector< vector< vector<int32> > > &all_path,
      Nnet &nnet, const vector<int> &randIndex, int pathN,
      const NnetDataRandomizerOptions& rnd_opts, bool crossvalidate,
      const string &target_model_filename, bool binary, bool randomize,
      const string &objective_function, int maxState){ 

    if (crossvalidate) {
      nnet.SetDropoutRetention(1.0);
    }

    RandomizerMask randomizer_mask(rnd_opts);
    MatrixRandomizer feature_randomizer(rnd_opts);
    PosteriorRandomizer targets_randomizer(rnd_opts);

    Xent xent;
    Mse mse;
    
    Matrix<BaseFloat> feats;
    CuMatrix<BaseFloat> nnet_out, obj_diff;

    Timer time;
    KALDI_LOG << (crossvalidate?"CROSS-VALIDATION":"TRAINING") << " STARTED";

    int32 num_done = 0, num_no_tgt_mat = 0, num_other_error = 0;
    kaldi::int64 total_frames = 0;

   
    for(int index = 0; index < randIndex.size(); ){
#if HAVE_CUDA==1
      // check the GPU is not overheated
      CuDevice::Instantiate().CheckGpuHealth();
#endif
      // fill the randomizer
      for ( ; index < randIndex.size(); index++) {
        if (feature_randomizer.IsFull()) break; // suspend, keep utt for next loop

        int utt_i = randIndex[index] / pathN;
        int path_i = randIndex[index] % pathN;

        assert(labfeats[utt_i].first.size() == all_path[utt_i][path_i].size());

        // get feature / target pair

        Posterior targets;
        makeFeature(feats, labfeats[utt_i], all_path[utt_i][path_i], maxState);
        makePost(targets, labfeats[utt_i].first, all_path[utt_i][path_i]);
        
        // pass data to randomizers
        KALDI_ASSERT(feats.NumRows() == targets.size());
        feature_randomizer.AddData(CuMatrix<BaseFloat>(feats));
        targets_randomizer.AddData(targets);
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
      }

      // train with data from randomizers (using mini-batches)
      for ( ; !feature_randomizer.Done(); feature_randomizer.Next(),
                                          targets_randomizer.Next()){
        // get block of feature/target pairs
        const CuMatrixBase<BaseFloat>& nnet_in = feature_randomizer.Value();
        const Posterior& nnet_tgt = targets_randomizer.Value();

        // forward pass
        nnet.Propagate(nnet_in, &nnet_out);

        // evaluate objective function we've chosen
        if (objective_function == "xent") {
          xent.Eval(nnet_out, nnet_tgt, &obj_diff);
        } else if (objective_function == "mse") {
          mse.Eval(nnet_out, nnet_tgt, &obj_diff);
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
}
