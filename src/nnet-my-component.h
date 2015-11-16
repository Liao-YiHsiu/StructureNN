#ifndef MY_COMPONENT_H_
#define MY_COMPONENT_H_

#include "nnet/nnet-component.h"
#include "nnet/nnet-utils.h"
#include "cudamatrix/cu-math.h"
#include "util/text-utils.h"
#include "util.h"

#include <algorithm>
#include <sstream>

using namespace std;

// an integration to add new Component to kaldi
class MyComponent : public UpdatableComponent{
   public:
      typedef enum {
         mUnknown = 0x0,

         mEmbed = 0x0100,
         mEmbedSimple,
         mEmbedMux,

         mBlend = 0x0200,
         mBlendSum
      } MyType;

      struct my_key_value{
         const MyType key;
         const char* value;
      };

      static const struct my_key_value myMarkerMap[];
      static const char*  myTypeToMarker(MyType t);
      static MyType       myMarkerToType(const string &s);

      static const char* myCompToMarker(const Component &comp);

      MyComponent(int32 input_dim, int32 output_dim): UpdatableComponent(input_dim, output_dim){}
      virtual ~MyComponent() {}
      virtual Component* Copy() const = 0;

      static Component* Init(const string &conf_line);
      static Component* Read(istream &is, bool binary);

      void Write(ostream &os, bool binary) const;

      virtual string Info() const { return ""; }
      virtual string InfoGradient() const { return ""; }

      // nerver called
      ComponentType GetType() const { return kUnknown; }
      virtual MyType myGetType() const = 0;

      virtual bool IsEmbed() const { return false;}
      virtual bool IsBlend() const { return false;}

      // make propagate and backpropagate virtual
      virtual void Propagate(const CuMatrixBase<BaseFloat> &in,
            CuMatrix<BaseFloat> *out){
         UpdatableComponent::Propagate(in, out);
      }

      virtual void Backpropagate(const CuMatrixBase<BaseFloat> &in,
            const CuMatrixBase<BaseFloat> &out,
            const CuMatrixBase<BaseFloat> &out_diff,
            CuMatrix<BaseFloat> *in_diff){
         UpdatableComponent::Backpropagate(in, out, out_diff, in_diff);
      }

   protected:
      virtual void PropagateFnc(const CuMatrixBase<BaseFloat> &in,
            CuMatrixBase<BaseFloat> *out) = 0;
      virtual void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in,
                         const CuMatrixBase<BaseFloat> &out,
                         const CuMatrixBase<BaseFloat> &out_diff,
                         CuMatrixBase<BaseFloat> *in_diff) = 0;

      // parameters...
      virtual void InitData(istream &is) = 0;
      virtual void ReadData(istream &is, bool binary) {}
      virtual void WriteData(ostream &os, bool binary) const {}

   protected:
      static MyComponent* NewMyComponentOfType(MyType type, int32 input_dim, int32 output_dim);
};

// Don't Resize Unless matrix size is too small.
class ComponentBuff : public Component{
   public:
      void Propagate(const CuMatrixBase<BaseFloat> &in, CuMatrix<BaseFloat> *out); 
      void Backpropagate(const CuMatrixBase<BaseFloat> &in,
            const CuMatrixBase<BaseFloat> &out,
            const CuMatrixBase<BaseFloat> &out_diff,
            CuMatrix<BaseFloat> *in_diff); 
};

#define NOT_UPDATABLE() \
      virtual bool IsUpdatable() const{ return false;} \
      virtual int32 NumParams() const{ return 0;} \
      virtual void GetParams(Vector<BaseFloat> *params) const { params->Resize(0); } \
      virtual void Update(const CuMatrixBase<BaseFloat> &input, \
            const CuMatrixBase<BaseFloat> &diff) {}

#endif
