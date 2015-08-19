#ifndef SNNET_H_
#define SNNET_H_

#include "nnet/nnet-nnet.h"
#include "base/kaldi-common.h"
#include "nnet/nnet-randomizer.h"
#include "util/common-utils.h"
#include "nnet-batch.h"
#include "util.h"
#include <iostream>
#include <string>
#include <vector>


using namespace std;
using namespace kaldi;
using namespace kaldi::nnet1;

class SNnet{
   public:
      SNnet():labelbuf_cols_(0) {}
      SNnet(const BNnet &nnet1, const BNnet &nnet2, uchar stateMax):
         nnet1_(nnet1), nnet2_(nnet2), stateMax_(stateMax), labelbuf_cols_(0) {}

      SNnet(const SNnet& other):
         nnet1_(other.nnet1_), nnet2_(other.nnet2_), stateMax_(other.stateMax_), labelbuf_cols_(0) {}

      SNnet &operator = (const SNnet& other); // Assignment operator.

      ~SNnet(); 

   public:
      /// NOTE: labels are 1-based not 0-based
      /// Perform forward pass through the network
      void Propagate(const vector<CuMatrix<BaseFloat>* > &in_arr,
            const vector<vector<uchar>* > &labels, CuMatrix<BaseFloat> *out); 

      /// Perform backward pass through the network
      void Backpropagate(const CuMatrix<BaseFloat> &out_diff, 
            const vector<vector<uchar>* >&labels);

      // compute f(x, lables) - f(ref_labels)
      void Propagate(const vector<CuMatrix<BaseFloat>* > &in_arr,
            const vector<vector<uchar>* > &labels,
            const vector<vector<uchar>* > &ref_labels, CuMatrix<BaseFloat> *out); 

      // backpropagate f(x, lables) - f(ref_labels)
      void Backpropagate(const vector<CuMatrix<BaseFloat> > &out_diff, 
            const vector<vector<uchar>* > &labels,
            const vector<vector<uchar>* > &ref_labels);
      
      /// Perform forward pass through the network, don't keep buffers 
      void Feedforward(const vector<CuMatrix<BaseFloat>* > &in_arr,
            const vector<vector<uchar>* > &labels, CuMatrix<BaseFloat> *out);

      /// speedup version for those in_arr are the same.
      void Feedforward(const CuMatrix<BaseFloat> &in,
            const vector<vector<uchar>* > &labels, CuMatrix<BaseFloat> *out);

      /// speedup version for those in_arr are the same.
      void Propagate(const CuMatrix<BaseFloat> &in,
            const vector<vector<uchar>* > &labels, CuMatrix<BaseFloat> *out);

      // Accumulate for cmvn
      void Acc(const CuMatrix<BaseFloat> &in, const vector<vector<uchar>* > &labels);

      // Get data from cmvn
      void Stat(CuVector<BaseFloat> &mean, CuVector<BaseFloat> &sd);

      // propagate to nnet1 only.
      void PropagatePsi(const CuMatrix<BaseFloat> &in,
            const vector<vector<uchar>* > &labels, CuMatrix<BaseFloat> *out);


      /// Dimensionality on network input (input feature dim.)
      int32 InputDim() const; 
      /// Dimensionality of network outputs (posteriors | bn-features | etc.)
      int32 OutputDim() const; 

      int32 NumParams() const;

      /// Set the dropout rate 
      void SetDropoutRetention(BaseFloat r);

      /// Initialize MLP from config
      void Init(const string &config_file1, const string &config_file2, uchar stateMax);
      /// Read the MLP from file (can add layers to exisiting instance of Nnet)
      void Read(const string &file1, const string &file2, uchar stateMax);  
      /// Write MLP to file
      void Write(const string &file1, const string &file2, bool binary) const;

      // Initialize only nnet1 (half network)
      void Read(const string &file1, uchar stateMax);

      /// Create string with human readable description of the nnet
      string Info() const;
      /// Create string with per-component gradient statistics
      string InfoGradient() const;
      /// Create string with propagation-buffer statistics
      string InfoPropagate() const;
      /// Create string with back-propagation-buffer statistics
      string InfoBackPropagate() const;

      /// Consistency check.
      void Check() const;
      /// Relese the memory
      void Destroy();

      /// Set training hyper-parameters to the network and its UpdatableComponent(s)
      /// TODO the learn rate of 2 nnets should be diff
      void SetTrainOptions(const NnetTrainOptions& opts, double ratio = 1);
      /// Get training hyper-parameters from the network
      const NnetTrainOptions& GetTrainOptions() const;

      void SetTransform(const Nnet &nnet);

   private:
      void Psi(vector<CuMatrix<BaseFloat> > &feats, const vector<vector<uchar>* > &labels, CuMatrix<BaseFloat> *out);
      void PsiKernel(vector<CuMatrix<BaseFloat> > &feats, const vector<vector<uchar>* > &labels, CuMatrix<BaseFloat> *out);
      int packPsi(vector<CuMatrix<BaseFloat> > &feats, const vector<vector<uchar>* > &labels, CuMatrix<BaseFloat> &psi_feats, CuVectorG<PsiPack> &packs_dev);
      void BackPsi(CuMatrix<BaseFloat> &diff, const vector<vector<uchar>* > &labels, vector<CuMatrix<BaseFloat> > &feats_diff);
      void BackPsiKernel(CuMatrix<BaseFloat> &diff, const vector<vector<uchar>* > &labels, vector<CuMatrix<BaseFloat> > &feats_diff);

      void makeFeat(CuMatrix<BaseFloat> &feat, const vector<uchar> &label, CuSubVector<BaseFloat> vec);
      void distErr(const CuSubVector<BaseFloat> &diff, const vector<uchar>& label, CuMatrix<BaseFloat> &mat);
      
      // size won't change. keep buff
      CuMatrix<BaseFloat> psi_buff_;
      CuMatrix<BaseFloat> psi_diff_;

      // size won't change. keep buff
      vector< CuMatrix<BaseFloat> > psi_arr_;
      vector< CuMatrix<BaseFloat> > out_arr_;

      vector< CuMatrix<BaseFloat> > psi_diff_arr_;

      vector<CuMatrix<BaseFloat> > transf_arr_;
      vector<CuMatrix<BaseFloat> > propagate_buf_;
      
      vector<CuMatrix<BaseFloat> > backpropagate_buf_;

      vector< vector<CuMatrix<BaseFloat> > > backpropagate_arr_;

      Nnet nnet_transf_;
      BNnet nnet1_;
      BNnet nnet2_;

      uchar stateMax_;

      // for kernel buffer
      vector<PsiPack>         packs_;
      CuVectorG<PsiPack>      packs_device_;
      vector<uchar>           labelbuf_; // 2-d array encode in 1-d
      int                     labelbuf_cols_;
      CuMatrixG<uchar>        labelbuf_device_;

      // for statistics
      CuMatrix<BaseFloat>  stat_sum_;
      CuMatrix<BaseFloat>  stat_sqr_;

      CuMatrix<BaseFloat>  stat_aux_;
      int                  stat_N_;

};

#endif // _SNNET_H_
