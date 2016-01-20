#ifndef MYNNET_H_
#define MYNNET_H_

#include "nnet/nnet-nnet.h"
#include "nnet/nnet-various.h"
#include "base/kaldi-common.h"
#include "nnet/nnet-randomizer.h"
#include "util/common-utils.h"
#include "bnnet.h"
#include "util.h"
#include "nnet-my-component.h"
#include <iostream>
#include <string>
#include <vector>


using namespace std;
using namespace kaldi;
using namespace kaldi::nnet1;

class MyNnet{
   public:
      MyNnet() : streamN_(1), labels_stride_(1), embedIdx_(-1), blendIdx_(-1) {}
      MyNnet(const MyNnet& other);
      MyNnet &operator = (const MyNnet &other);

      ~MyNnet() { Destroy(); }

   public:
      void Propagate(const CuMatrixBase<BaseFloat> &in, CuMatrix<BaseFloat> *out);

      void Backpropagate(const CuMatrixBase<BaseFloat> &out_diff, CuMatrix<BaseFloat> *in_diff);

      int32 InputDim() const;
      int32 OutputDim() const;

      int32 NumComponents() const {return components_.size(); }

      const Component& GetComponent(int32 c) const;
      Component& GetComponent(int32 c);

      void SetComponent(int32 c, Component *comp);

      void AppendComponent(Component *comp);

      int32 NumParams() const;

      void GetParams(Vector<BaseFloat> *weight) const;

      void SetDropoutRetention(BaseFloat r);

      int32 GetLabelNum() const;

      void SetLabelSeqs(const vector<int32> &labels, int labels_stride);

      void ResetLstmStreams(const vector<int32> &stream_reset_flag);

      void SetSeqLengths(const vector<int32> &sequence_lengths);

      void SetBuff(int max_input_rows, int labels_stride, int streamN);

      void Init(const string& config_file);
      
      void Read(const string& file);

      void Read(istream &is, bool binary);

      void Write(const string &file, bool binary) const;
      
      void Write(ostream &os, bool binary) const;

      string Info() const;

      string InfoGradient() const;

      string InfoPropagate() const;

      string InfoBackPropagate() const;

      void Check() const;

      void Destroy();

      void SetTrainOptions(const NnetTrainOptions& opts);

      void forceBlend();

   private:

      vector<Component*> components_;
      
      vector< CuMatrix<BaseFloat> > propagate_buf_;
      vector< CuMatrix<BaseFloat> > backpropagate_buf_;

      vector< int > rows_num_;

      NnetTrainOptions opts_;

      int streamN_;
      int labels_stride_;
      int embedIdx_; // there's should be only one or less embedding layer
      int blendIdx_; // there's should be only one or less blending layer
};

#endif
