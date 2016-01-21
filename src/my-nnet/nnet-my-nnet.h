#ifndef MYNNET_H_
#define MYNNET_H_

#include "base/kaldi-common.h"
#include "util/common-utils.h"

#include "my-utils/util.h"
#include "my-nnet/nnet-my-component.h"
#include <iostream>
#include <string>
#include <vector>


using namespace std;
using namespace kaldi;

class MyNnet{
   public:
      MyNnet() : streamN_(1), labels_stride_(1), embedIdx_(-1), blendIdx_(-1) {}
      MyNnet(const MyNnet& other);
      MyNnet &operator = (const MyNnet &other);

      ~MyNnet() { Destroy(); }

   public:
      void Propagate(const CuMatrixBase<BaseFloat> &in, MyCuMatrix<BaseFloat> *out);

      // only do backpropagte no update...
      void Backpropagate(const CuMatrixBase<BaseFloat> &out_diff, MyCuMatrix<BaseFloat> *in_diff);
      // update accumulated gradient.
      void Update();

      void Feedforward(const CuMatrixBase<BaseFloat> &in, MyCuMatrix<BaseFloat> *out);

      int32 InputDim() const;
      int32 OutputDim() const;

      int32 NumComponents() const {return components_.size(); }

      const MyComponent& GetComponent(int32 c) const;
      MyComponent& GetComponent(int32 c);

      void SetComponent(int32 c, MyComponent *comp);

      void AppendComponent(MyComponent *comp);

      void AppendNnet(const MyNnet& nent_to_append);

      void RemoveComponent(int32 c);
      void RemoveLastComponent() { RemoveComponent(NumComponents() - 1); }

      const vector< MyCuMatrix<BaseFloat> >& PropagateBuffer() const{
         return propagate_buf_;
      }

      const vector< MyCuMatrix<BaseFloat> >& BackpropagateBuffer() const {
         return backpropagate_buf_;
      }

      int32 NumParams() const;

      void GetParams(Vector<BaseFloat> *weight) const;

      // for Dropout
      //void SetDropoutRetention(BaseFloat r);

      // LSTM
      void ResetLstmStreams(const vector<int32> &stream_reset_flag);

      // BSLTM
      void SetSeqLengths(const vector<int32> &sequence_lengths);

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
      
      const NnetTrainOptions& GetTrainOptions() const{
         return opts_;
      }

   private:

      vector<MyComponent*> components_;
      
      vector< MyCuMatrix<BaseFloat> > propagate_buf_;
      vector< MyCuMatrix<BaseFloat> > backpropagate_buf_;

      NnetTrainOptions opts_;

// <structureLearning>
   public:
      int32 GetLabelNum() const;
      void SetLabelSeqs(const vector<uchar> &labels, int labels_stride);
      void forceBlend();
   private:
      int streamN_;
      int labels_stride_;
      int embedIdx_; // there's should be only one or less embedding layer
      int blendIdx_; // there's should be only one or less blending layer
      vector< int > rows_num_;
// </structureLearning>
};

#endif
