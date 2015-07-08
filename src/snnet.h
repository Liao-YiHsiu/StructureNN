#include "nnet/nnet-nnet.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "nnet-cache.h"
#include <iostream>
#include <string>
#include <vector>

#ifndef SNNET_H_
#define SNNET_H_

using namespace std;
using namespace kaldi::nnet1;

class SNnet{
   public:
      SNnet() {}
      SNnet(const CNnet &nnet1, const Nnet &nnet2, int stateMax):
         nnet1_(nnet1), nnet2_(nnet2), stateMax_(stateMax) {}

      SNnet(const SNnet& other):
         nnet1_(other.nnet1_), nnet2_(other.nnet2_), stateMax_(other.stateMax_) {}

      SNnet &operator = (const SNnet& other); // Assignment operator.

      ~SNnet(); 

   public:
      /// NOTE: labels are 1-based not 0-based
      /// Perform forward pass through the network
      void Propagate(const vector<CuMatrix<BaseFloat>* > &in_arr, const vector<vector<int32> > &labels, CuMatrix<BaseFloat> *out); 
      /// Perform backward pass through the network
      void Backpropagate(const CuMatrix<BaseFloat> &out_diff);
      /// Perform forward pass through the network, don't keep buffers (use it when not training)
      void Feedforward(const vector<CuMatrix<BaseFloat>* > &in_arr, const vector<vector<int32> > &labels, CuMatrix<BaseFloat> *out);

      /// Dimensionality on network input (input feature dim.)
      int32 InputDim() const; 
      /// Dimensionality of network outputs (posteriors | bn-features | etc.)
      int32 OutputDim() const; 

      int32 NumParams() const;

      /// Initialize MLP from config
      void Init(const string &config_file1, const string &config_file2, int stateMax);
      /// Read the MLP from file (can add layers to exisiting instance of Nnet)
      void Read(const string &file1, const string &file2, int stateMax);  
      /// Write MLP to file
      void Write(const string &file1, const string &file2, bool binary) const;

      /// Create string with human readable description of the nnet
      string Info() const;
      /// Create string with per-component gradient statistics
      string InfoGradient() const;
      /// Consistency check.
      void Check() const;
      /// Relese the memory
      void Destroy();

      /// Set training hyper-parameters to the network and its UpdatableComponent(s)
      /// TODO the learn rate of 2 nnets should be diff
      void SetTrainOptions(const NnetTrainOptions& opts);
      /// Get training hyper-parameters from the network
      const NnetTrainOptions& GetTrainOptions() const;

   private:
      void Psi(vector<CuMatrix<BaseFloat> > &feats, const vector<vector<int32> > &labels, CuMatrix<BaseFloat> *out);
      void BackPsi(const CuMatrix<BaseFloat> &diff, const vector<vector<int32> > &labels, vector<CuMatrix<BaseFloat> > *feats_diff);

      void makeFeat(CuMatrix<BaseFloat> &feat, const vector<int32> &label, CuSubVector<BaseFloat> &vec);
      void distErr(const CuSubVector<BaseFloat> &diff, const vector<int32>& label, CuMatrix<BaseFloat> &mat);

      vector<CuMatrix<BaseFloat> > propagate_buf_;
      vector<CuMatrix<BaseFloat> > backpropagate_buf_;

      vector< vector<int32> > labels_;

      CuMatrix<BaseFloat> psi_buff_;
      CuMatrix<BaseFloat> psi_diff_;

      CNnet nnet1_;
      Nnet nnet2_;

      int stateMax_;
};

#endif // _SNNET_H_
