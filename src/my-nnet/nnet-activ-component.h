#ifndef ACTIV_COMPONENT_H_
#define ACTIV_COMPONENT_H_

#include "nnet/nnet-component.h"
#include "nnet/nnet-utils.h"
#include "cudamatrix/cu-math.h"
#include "util/text-utils.h"
#include "util.h"

#include "nnet-my-component.h"

#include <algorithm>
#include <sstream>

using namespace std;

class Activ : public MyComponent{
   public:
      Activ(int32 input_dim, int32 output_dim):
         MyComponent(input_dim, output_dim){} 
      virtual ~Activ() {}

      virtual void InitData(istream &is) {}
      virtual void ReadData(istream &is, bool binary){}
      virtual void WriteData(ostream &os, bool binary) const {}
};

class ReLU : public Activ{
   public:
      ReLU(int32 input_dim, int32 output_dim):
         Activ(input_dim, output_dim){}
      virtual ~ReLU() {}

      Component* Copy() const;

      MyType myGetType() const { return mReLU; }

      NOT_UPDATABLE();

   protected:
      void PropagateFnc(const CuMatrixBase<BaseFloat> &in,
            CuMatrixBase<BaseFloat> *out);
      void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in,
            const CuMatrixBase<BaseFloat> &out,
            const CuMatrixBase<BaseFloat> &out_diff,
            CuMatrixBase<BaseFloat> *in_diff);
};
#endif
