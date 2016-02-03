#ifndef ACTIV_COMPONENT_H_
#define ACTIV_COMPONENT_H_

#include "nnet/nnet-component.h"
#include "nnet/nnet-utils.h"
#include "cudamatrix/cu-math.h"
#include "util/text-utils.h"

#include "my-utils/util.h"
#include "my-nnet/nnet-my-component.h"

#include <algorithm>
#include <sstream>

using namespace std;

class Activ : public MyComponent{
   public:
      Activ(int32 input_dim, int32 output_dim):
         MyComponent(input_dim, output_dim){} 
      virtual ~Activ() {}

      NOT_UPDATABLE();
      virtual void InitData(istream &is) {}
      virtual void ReadData(istream &is, bool binary){}
      virtual void WriteData(ostream &os, bool binary) const {}
};

class myReLU : public Activ{
   public:
      myReLU(int32 input_dim, int32 output_dim):
         Activ(input_dim, output_dim){}
      virtual ~myReLU() {}

      MyComponent* Copy() const { return new myReLU(input_dim_, output_dim_); }

      MyType GetType() const { return mReLU; }

   protected:
      void PropagateFnc(const CuMatrixBase<BaseFloat> &in,
            CuMatrixBase<BaseFloat> *out){
         out->CopyFromMat(in);
         out->ApplyFloor(0.0);
      }

      void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in,
            const CuMatrixBase<BaseFloat> &out,
            const CuMatrixBase<BaseFloat> &out_diff,
            CuMatrixBase<BaseFloat> *in_diff){
         in_diff->CopyFromMat(out);
         in_diff->ApplyHeaviside();
         in_diff->MulElements(out_diff);
      }
};

class mySoftmax : public Activ{
   public:
      mySoftmax(int32 dim_in, int32 dim_out) : 
         Activ(dim_in, dim_out) { }
      virtual ~mySoftmax() { }

      MyComponent* Copy() const { return new mySoftmax(*this); }

      MyType GetType() const { return mSoftmax; }

   protected:
      void PropagateFnc(const CuMatrixBase<BaseFloat> &in, CuMatrixBase<BaseFloat> *out) {
         out->ApplySoftMaxPerRow(in);
      }

      void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in, const CuMatrixBase<BaseFloat> &out,
            const CuMatrixBase<BaseFloat> &out_diff, CuMatrixBase<BaseFloat> *in_diff) {
         in_diff->CopyFromMat(out_diff);
      }
};

class mySigmoid : public Activ {
   public:
      mySigmoid(int32 dim_in, int32 dim_out) : 
         Activ(dim_in, dim_out) { }
      ~mySigmoid() { }

      MyComponent* Copy() const { return new mySigmoid(*this); }
      MyType GetType() const { return mSigmoid; }

   protected:
      void PropagateFnc(const CuMatrixBase<BaseFloat> &in, CuMatrixBase<BaseFloat> *out) {
         out->Sigmoid(in);
      }

      void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in, const CuMatrixBase<BaseFloat> &out,
            const CuMatrixBase<BaseFloat> &out_diff, CuMatrixBase<BaseFloat> *in_diff) {
         in_diff->DiffSigmoid(out, out_diff);
      }
};

class myTanh : public Activ {
   public:
      myTanh(int32 dim_in, int32 dim_out) :
         Activ(dim_in, dim_out) { }
      ~myTanh() { }

      MyComponent* Copy() const { return new myTanh(*this); }
      MyType GetType() const { return mTanh; }

   protected:
      void PropagateFnc(const CuMatrixBase<BaseFloat> &in, CuMatrixBase<BaseFloat> *out) {
         out->Tanh(in);
      }

      void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in, const CuMatrixBase<BaseFloat> &out,
            const CuMatrixBase<BaseFloat> &out_diff, CuMatrixBase<BaseFloat> *in_diff) {
         in_diff->DiffTanh(out, out_diff);
      }
};

class myDropout : public Activ {
   public:
      myDropout(int32 dim_in, int32 dim_out):
         Activ(dim_in, dim_out), dropout_retention_(0.5) { }
      ~myDropout() { }

      MyComponent* Copy() const { return new myDropout(*this); }
      MyType GetType() const { return mDropout; }

      void InitData(istream &is) {
         is >> ws; // eat-up whitespace
         string token; 
         while (!is.eof()) {
            ReadToken(is, false, &token); 
            /**/ if (token == "<DropoutRetention>") ReadBasicType(is, false, &dropout_retention_);
            else KALDI_ERR << "Unknown token " << token << ", a typo in config?"
               << " (DropoutRetention)";
         }
         KALDI_ASSERT(dropout_retention_ > 0.0 && dropout_retention_ <= 1.0);
      }

      void ReadData(istream &is, bool binary) {
         if ('<' == Peek(is, binary)) {
            ExpectToken(is, binary, "<DropoutRetention>");
            ReadBasicType(is, binary, &dropout_retention_);
         }
         KALDI_ASSERT(dropout_retention_ > 0.0 && dropout_retention_ <= 1.0);
      }

      void WriteData(ostream &os, bool binary) const {
         WriteToken(os, binary, "<DropoutRetention>");
         WriteBasicType(os, binary, dropout_retention_);
      }

   protected:
      void PropagateFnc(const CuMatrixBase<BaseFloat> &in, CuMatrixBase<BaseFloat> *out) {
         out->CopyFromMat(in);
         // switch off 50% of the inputs...
         dropout_mask_.Resize(out->NumRows(),out->NumCols());
         dropout_mask_.Set(dropout_retention_);
         rand_.BinarizeProbs(dropout_mask_,&dropout_mask_);
         out->MulElements(dropout_mask_);
         // rescale to keep same dynamic range as w/o dropout
         out->Scale(1.0/dropout_retention_);
      }

      void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in, const CuMatrixBase<BaseFloat> &out,
            const CuMatrixBase<BaseFloat> &out_diff, CuMatrixBase<BaseFloat> *in_diff) {
         in_diff->CopyFromMat(out_diff);
         // use same mask on the error derivatives...
         in_diff->MulElements(dropout_mask_);
         // enlarge output to fit dynamic range w/o dropout
         in_diff->Scale(1.0/dropout_retention_);
      }

   public:
      BaseFloat GetDropoutRetention() {
         return dropout_retention_;
      }

      void SetDropoutRetention(BaseFloat dr) {
         dropout_retention_ = dr;
         KALDI_ASSERT(dropout_retention_ > 0.0 && dropout_retention_ <= 1.0);
      }

   private:
      CuRand<BaseFloat> rand_;
      CuMatrix<BaseFloat> dropout_mask_;
      BaseFloat dropout_retention_;
};


#endif
