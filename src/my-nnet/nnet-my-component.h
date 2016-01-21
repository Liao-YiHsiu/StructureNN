#ifndef MY_COMPONENT_H_
#define MY_COMPONENT_H_

#include "base/kaldi-common.h"
#include "cudamatrix/cu-matrix.h"
#include "util/text-utils.h"

#include "my-utils/util.h"
#include "my-cumatrix/cu-matrix.h"
#include "my-cumatrix/cu-general.h"
#include "my-cumatrix/cu-utils.h"
#include "my-cuda-kernel/kernel-func.h"

#include <algorithm>
#include <sstream>

using namespace std;
using namespace kaldi;

class MyComponent {
   public:
      typedef enum {
         mUnknown = 0x0,

         mEmbed = 0x0100,
         mEmbedSimple,

         mBlend = 0x0200,
         mBlendSum,

         mActiv = 0x0300,
         mReLU,

         mLayer = 0x0400,
         mLSTM,
         mAffine,

      } MyType;

      struct key_value{
         const MyType key;
         const char* value;
      };

      static const struct key_value MarkerMap[];
      static const char*  TypeToMarker(MyType t);
      static MyType       MarkerToType(const string &s);

   public:
      MyComponent(int32 input_dim, int32 output_dim):input_dim_(input_dim), output_dim_(output_dim){}
      virtual ~MyComponent() {}

      virtual MyComponent* Copy() const = 0;

      virtual MyType GetType() const = 0;

      int32 InputDim() const { return input_dim_; }

      int32 OutputDim() const { return output_dim_; }

      // make propagate and backpropagate virtual
      virtual void Propagate(const CuMatrixBase<BaseFloat> &in,
            MyCuMatrix<BaseFloat> *out);

      virtual void Backpropagate(const CuMatrixBase<BaseFloat> &in,
            const CuMatrixBase<BaseFloat> &out,
            const CuMatrixBase<BaseFloat> &out_diff,
            MyCuMatrix<BaseFloat> *in_diff);

      virtual void Update() = 0;

      static MyComponent* Init(const string &conf_line);
      static MyComponent* Read(istream &is, bool binary);

      void Write(ostream &os, bool binary) const;

      virtual string Info() const { return ""; }
      virtual string InfoGradient() const { return ""; }

      // <structured> 
      virtual bool IsEmbed() const { return false;}
      virtual bool IsBlend() const { return false;}
      // </structured>

   protected:
      virtual void PropagateFnc(const CuMatrixBase<BaseFloat> &in,
            CuMatrixBase<BaseFloat> *out) = 0;
      virtual void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in,
                         const CuMatrixBase<BaseFloat> &out,
                         const CuMatrixBase<BaseFloat> &out_diff,
                         CuMatrixBase<BaseFloat> *in_diff) = 0;

      // parameters...
      virtual void ReadData(istream &is, bool binary) {}
      virtual void WriteData(ostream &os, bool binary) const {}

      int32 input_dim_;
      int32 output_dim_;

   protected:
      static MyComponent* NewMyComponentOfType(MyType type, int32 input_dim, int32 output_dim);

      // --------------------------------------------------------------
      //<UpdatableComponent>
   public:
      bool IsUpdatable() const { return true; }
      virtual int32 NumParams() const = 0;
      virtual void GetParams(Vector<BaseFloat> *params) const = 0;
      virtual void Update(const CuMatrixBase<BaseFloat> &input, 
            const CuMatrixBase<BaseFloat> &diff) = 0;
      const NnetTrainOptions& GetTrainOptions() const{
         return opts_;
      }
      virtual void SetTrainOptions(const NnetTrainOptions& opts){
         opts_ = opts;
      }
      virtual void InitData(istream &is) = 0;
   protected:
      NnetTrainOptions opts_;
      //</UpdatableComponent>
};

#define NOT_UPDATABLE() \
      virtual bool IsUpdatable() const{ return false;} \
      virtual int32 NumParams() const{ return 0;} \
      virtual void GetParams(Vector<BaseFloat> *params) const { params->Resize(0); } \
      virtual void Update(const CuMatrixBase<BaseFloat> &input, \
            const CuMatrixBase<BaseFloat> &diff) {} \
      virtual void Update() {}

#endif
