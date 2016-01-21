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

      NOT_UPDATABLE();

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
#endif
