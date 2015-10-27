#ifndef SRNNET_H_
#define SRNNET_H_

#include "nnet/nnet-nnet.h"
#include "nnet/nnet-various.h"
#include "base/kaldi-common.h"
#include "nnet/nnet-randomizer.h"
#include "util/common-utils.h"
#include "bnnet.h"
#include "util.h"
#include <iostream>
#include <string>
#include <vector>


using namespace std;
using namespace kaldi;
using namespace kaldi::nnet1;

class SRNnet2{
   public:
      SRNnet2(): rnn_init_(NULL), forw_component_(NULL), acti_component_(NULL){}
      SRNnet2(const SRNnet2& other);
      SRNnet2 &operator = (const SRNnet2& other); // Assignment operator.

      ~SRNnet2(); 

   public:
      /// NOTE: labels are 1-based not 0-based
      /// Perform forward pass through the network
      void Propagate(const CuMatrix<BaseFloat> &in,
            const vector<vector<uchar>* > &labels, CuMatrix<BaseFloat> *out);

      /// Perform backward pass through the network
      void Backpropagate(const CuMatrix<BaseFloat> &out_diff, 
            const vector<vector<uchar>* >&labels);

      // frameshuffle
      void Propagate(const CuMatrix<BaseFloat> &in,
            const vector<uchar> &label, int pos, CuMatrix<BaseFloat> *out);

      void Backpropagate(const CuMatrix<BaseFloat> &out_diff, 
            const vector<uchar> &labels, int pos, int depth);

      /// speedup version for those in_arr are the same.
      //void Feedforward(const CuMatrix<BaseFloat> &in,
      //      const vector<vector<uchar>* > &labels, CuMatrix<BaseFloat> *out);

      void Decode(const CuMatrix<BaseFloat> &in, ScorePath::Table &table, int Nbest);

      /// Dimensionality on network input (input feature dim.)
      int32 InputDim() const; 
      /// Dimensionality of network outputs (posteriors | bn-features | etc.)
      int32 OutputDim() const; 

      uchar StateMax() const;

      /// Get the number of parameters in the network
      int32 NumParams() const;
      /// Get the network weights in a supervector
      void GetParams(Vector<BaseFloat>* wei_copy) const;
      /// Get the network weights in a supervector
      void GetWeights(Vector<BaseFloat>* wei_copy) const;
      /// Set the network weights from a supervector
      void SetWeights(const Vector<BaseFloat>& wei_src);
      /// Get the gradient stored in the network
      void GetGradient(Vector<BaseFloat>* grad_copy) const;

      /// Set the dropout rate 
      void SetDropoutRetention(BaseFloat r);

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
      void SetTrainOptions(const NnetTrainOptions& opts, double ratio = 1);

      void SetTransform(const Nnet &nnet);

   private:
      void PropagateRPsi(const CuMatrix<BaseFloat> &in, const vector< vector<uchar>* > &labels);

      void RPsi(vector< CuMatrix<BaseFloat> > &propagate_phone,
            const vector< vector<uchar>* > &labels, vector< CuMatrix<BaseFloat> > &propagate_frame);

      void BackRPsi(vector< CuMatrix<BaseFloat> > &backpropagate_frame,
            const vector< vector<uchar>* > &labels, vector< CuMatrix<BaseFloat> > &backpropagate_phone);

      void packRPsi(vector< CuMatrix<BaseFloat> > &phone_mat,
            const vector< vector<uchar>* > &labels, vector< CuMatrix<BaseFloat> > &frame_mat,
            RPsiPack* pack);


      string Details(const Component* comp)const;
      
      Nnet nnet_transf_;
      Nnet nnet1_;
      BNnet nnet2_;
      
      uchar stateMax_;

      AddShift*                    rnn_init_;
      vector<UpdatableComponent* > mux_components_;

      UpdatableComponent*          forw_component_;
      Component*                   acti_component_;

      // TODO for bidirectional RNN
      //AffineTransform* back_component_;

      // -------------------- buf -------------------------
      CuMatrix<BaseFloat>           transf_;
      CuMatrix<BaseFloat>           propagate_feat_buf_;
      CuMatrix<BaseFloat>           propagate_init_;

      vector< CuMatrix<BaseFloat> > propagate_phone_buf_;

      vector< CuMatrix<BaseFloat> > propagate_frame_buf_;
      vector< CuMatrix<BaseFloat> > propagate_acti_in_buf_;
      vector< CuMatrix<BaseFloat> > propagate_acti_out_buf_;
      vector< CuMatrix<BaseFloat> > propagate_forw_buf_;
      vector< CuMatrix<BaseFloat> > propagate_score_buf_;

      vector< CuMatrix<BaseFloat> > backpropagate_acti_out_buf_;
      vector< CuMatrix<BaseFloat> > backpropagate_acti_in_buf_;
      vector< CuMatrix<BaseFloat> > backpropagate_forw_buf_;

      vector< CuMatrix<BaseFloat> > backpropagate_phone_buf_;
      vector< CuMatrix<BaseFloat> > backpropagate_feat_buf_;

      CuMatrix<BaseFloat>           backpropagate_all_feat_buf_;

      // for Propagate label
      CuMatrix<BaseFloat>           propagate_phone_;
      CuMatrix<BaseFloat>           propagate_acti_out_;
      CuMatrix<BaseFloat>           propagate_feat_;

      CuMatrix<BaseFloat>           backpropagate_acti_out_;
      CuMatrix<BaseFloat>           backpropagate_acti_in_;
      CuMatrix<BaseFloat>           backpropagate_forw_;
      CuMatrix<BaseFloat>           backpropagate_feat_;
      CuMatrix<BaseFloat>           backpropagate_all_feat_;

      CuMatrix<BaseFloat>           propagate_phone_tmp_;
      CuMatrix<BaseFloat>           backpropagate_feat_tmp_;
      CuMatrix<BaseFloat>           backpropagate_all_feat_tmp_;


      // double buffer for decoding
      vector< vector< CuMatrix<BaseFloat> > > double_buffer_;
      CuMatrix<BaseFloat>                     propagate_acti_in_;
      CuMatrix<BaseFloat>                     propagate_forw_;
      CuMatrix<BaseFloat>                     propagate_score_;
      Matrix<BaseFloat>                       propagate_score_host_;

      
      // kernel buffer
      vector<uchar>         labelbuf_;
      CuMatrixG<uchar>      labelbuf_dev_;

      vector<BaseFloat*>    phone_mat_pt_;
      CuVectorG<BaseFloat*> phone_mat_pt_dev_;

      vector<BaseFloat*>    frame_mat_pt_;
      CuVectorG<BaseFloat*> frame_mat_pt_dev_;


};

typedef struct{
   uchar       phone;
   BaseFloat   score;
   int         bt;  // back track
} Token;

bool compareToken(const Token &a, const Token &b);

#endif // _SNNET_H_
