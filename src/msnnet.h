#ifndef MSNNET_H_
#define MSNNET_H_

#include "nnet/nnet-nnet.h"
#include "nnet/nnet-various.h"
#include "base/kaldi-common.h"
#include "nnet/nnet-randomizer.h"
#include "util/common-utils.h"
#include "bnnet.h"
#include "nnet-mux-component.h"
#include "util.h"
#include <iostream>
#include <string>
#include <vector>


using namespace std;
using namespace kaldi;
using namespace kaldi::nnet1;

// mux implemented structure NN
class MSNnet{
   public:
      MSNnet(): mux_(NULL), streamN_(1){}
      MSNnet(const MSNnet& other);
      MSNnet &operator = (const MSNnet& other); // Assignment operator.

      ~MSNnet(); 

   public:
      void Propagate(const CuMatrixBase<BaseFloat> &in, CuMatrix<BaseFloat> *out);

      void Backpropagate(const CuMatrixBase<BaseFloat> &out_diff, CuMatrix<BaseFloat> *in_diff);

      void Feedforward(const CuMatrixBase<BaseFloat> &in, CuMatrix<BaseFloat> *out);

      /// Dimensionality on network input (input feature dim.)
      int32 InputDim() const { return nnet1_.InputDim(); }
      /// Dimensionality of network outputs (posteriors | bn-features | etc.)
      int32 OutputDim() const { return nnet2_.OutputDim(); }

      void SetNnet(const Nnet &nnet1, const Nnet &nnet2, const Mux &mux);
      const Nnet& GetNnet1() { return nnet1_; }
      const Nnet& GetNnet2() { return nnet2_; }
      const Mux&  GetMux()   { return *mux_; }

      /// Get the number of parameters in the network
      int32 NumParams() const;

      int StateMax() const { return mux_->NumComponents(); }

      /// Set the dropout rate 
      void SetDropoutRetention(BaseFloat r);

      void SetLabelSeqs(const vector<int32> &labels, int labels_stride);

      // required to set labels first.
      // for lstm
      void ResetLstmStreams(const vector<int32> &stream_reset_flag);
      // for blstm
      void SetSeqLengths(const vector<int32> &sequence_lengths);

      /// Initialize MLP from config
      void Init(const Nnet& nnet1, const Nnet& nnet2, const string &config_file);

      /// Read the MLP from file (can add layers to exisiting instance of Nnet)
      void Read(const string &file);  
      void Read(istream &is, bool binary);
      /// Write MLP to file
      void Write(const string &file, bool binary) const;
      void Write(ostream &os, bool binary) const;

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
      void SetTrainOptions(const NnetTrainOptions& opts);

   private:
      vector<int32> expandVec(const vector<int32> &src, int mul);

      Nnet nnet1_;
      Nnet nnet2_;
      Mux* mux_;

      CuMatrix<BaseFloat> propagate_nnet1_out_buf_;
      CuMatrix<BaseFloat> propagate_nnet2_out_buf_;
      CuMatrix<BaseFloat> propagate_mux_out_buf_;

      CuMatrix<BaseFloat> backpropagate_nnet2_in_buf_;
      CuMatrix<BaseFloat> backpropagate_mux_in_buf_;
      CuMatrix<BaseFloat> backpropagate_out_buf_;

      int streamN_;

      vector<int32> labels_;
      int labels_stride_;

};

#endif // _SNNET_H_
