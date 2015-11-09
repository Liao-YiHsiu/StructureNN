#ifndef BLEND_COMPONENT_H_
#define BLEND_COMPONENT_H_

#include "nnet/nnet-component.h"
#include "nnet/nnet-utils.h"
#include "cudamatrix/cu-math.h"
#include "util/text-utils.h"
#include "util.h"

#include "nnet-my-component.h"

#include <algorithm>
#include <sstream>

using namespace std;

class Blend : public MyComponent{
   public:
      Blend(int32 input_dim, int32 output_dim):
         MyComponent(input_dim, output_dim){ assert(input_dim == output_dim); }
      virtual ~Blend() {}

      bool IsBlend() const { return true; }

      void Propagate(const CuMatrixBase<BaseFloat> &in,
            CuMatrix<BaseFloat> *out);
      void Backpropagate(const CuMatrixBase<BaseFloat> &in,
            const CuMatrixBase<BaseFloat> &out,
            const CuMatrixBase<BaseFloat> &out_diff,
            CuMatrix<BaseFloat> *in_diff);

      virtual void InitData(istream &is) {}
      virtual void ReadData(istream &is, bool binary){}
      virtual void WriteData(ostream &os, bool binary) const {}

      void SetSeqLengths(const vector<int32> &seq_length);

   protected:
      virtual void SetSeqLengthsFnc(const vector<int32> &seq_length){}

      vector<int32> seq_length_;
      int32         max_length_;
};

class BlendSum : public Blend{
   public:
      BlendSum(int32 input_dim, int32 output_dim):
         Blend(input_dim, output_dim){}
      ~BlendSum() {}

      Component* Copy() const;

      MyType myGetType() const { return mBlendSum; }

      NOT_UPDATABLE();

   protected:
      void PropagateFnc(const CuMatrixBase<BaseFloat> &in,
            CuMatrixBase<BaseFloat> *out);
      void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in,
            const CuMatrixBase<BaseFloat> &out,
            const CuMatrixBase<BaseFloat> &out_diff,
            CuMatrixBase<BaseFloat> *in_diff);

      void SetSeqLengthsFnc(const vector<int32> &seq_length);

   private:
      CuVectorG<int32> seq_device_;
};
#endif
