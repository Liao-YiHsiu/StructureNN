#ifndef BLEND_COMPONENT_H_
#define BLEND_COMPONENT_H_

#include "nnet/nnet-component.h"
#include "nnet/nnet-utils.h"
#include "cudamatrix/cu-math.h"
#include "util/text-utils.h"

#include "my-nnet/nnet-my-component.h"

#include <algorithm>
#include <sstream>

using namespace std;

class Blend : public MyComponent{
   public:
      Blend(int32 input_dim, int32 output_dim):
         MyComponent(input_dim, output_dim){ assert(input_dim == output_dim); }
      virtual ~Blend() {}

      virtual bool IsBlend() const { return true; }

      virtual void Propagate(const CuMatrixBase<BaseFloat> &in,
            MyCuMatrix<BaseFloat> *out){
         assert( input_dim_ == in.NumCols() );

         out->Resize(seq_length_.size(), output_dim_);
         PropagateFnc(in, out);
      }

      virtual void Backpropagate(const CuMatrixBase<BaseFloat> &in,
            const CuMatrixBase<BaseFloat> &out,
            const CuMatrixBase<BaseFloat> &out_diff,
            MyCuMatrix<BaseFloat> *in_diff){
         assert( input_dim_ == in.NumCols() );
         assert( output_dim_ == out.NumCols() && out.NumRows() == seq_length_.size() );
         assert( out_diff.NumCols() == out.NumCols() && out_diff.NumRows() == out.NumRows());

         if(in_diff == NULL) return;

         in_diff->Resize(in.NumRows(), in.NumCols());
         BackpropagateFnc(in, out, out_diff, in_diff);
      }

      virtual void InitData(istream &is) {}
      virtual void ReadData(istream &is, bool binary){}
      virtual void WriteData(ostream &os, bool binary) const {}

      void SetSeqLengths(const vector<int32> &seq_length){
         seq_length_ = seq_length;
         SetSeqLengthsFnc(seq_length);
      }

   protected:
      virtual void SetSeqLengthsFnc(const vector<int32> &seq_length) = 0;

      vector<int32> seq_length_;
};

class BlendSum : public Blend{
   public:
      BlendSum(int32 input_dim, int32 output_dim):
         Blend(input_dim, output_dim){}
      ~BlendSum() {}

      MyComponent* Copy() const{
         return new BlendSum(input_dim_, output_dim_);
      }

      MyType GetType() const { return mBlendSum; }

      NOT_UPDATABLE();

   protected:
      void PropagateFnc(const CuMatrixBase<BaseFloat> &in,
            CuMatrixBase<BaseFloat> *out){
         blendsum_prop(in, seq_device_.Data(), seq_length_.size(), *out);
         // check consistence
         //CuMatrix<BaseFloat> tmp_out(seq_length_.size(), output_dim_, kSetZero);
         //for(int i = 0; i < in.NumRows(); ++i){
         //   int t   = i / seq_length_.size();
         //   int idx = i % seq_length_.size();
         //   if(t < seq_length_[idx]){
         //      tmp_out.Row(idx).AddVec(1.0, in.Row(i));
         //   }
         //}
         //assert(Same(*out, tmp_out));
      }

      void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in,
            const CuMatrixBase<BaseFloat> &out,
            const CuMatrixBase<BaseFloat> &out_diff,
            CuMatrixBase<BaseFloat> *in_diff){
         blendsum_back(out_diff, seq_device_.Data(), seq_length_.size(), *in_diff);
         // check consistence
         //CuMatrix<BaseFloat> tmp_in_diff(in.NumRows(), input_dim_, kSetZero);
         //for(int i = 0; i < tmp_in_diff.NumRows(); ++i){
         //   int t   = i / seq_length_.size();
         //   int idx = i % seq_length_.size();
         //   if(t < seq_length_[idx]){
         //      tmp_in_diff.Row(i).CopyFromVec(out_diff.Row(idx));
         //   }
         //}
         //assert(Same(*in_diff, tmp_in_diff));
      }

      void SetSeqLengthsFnc(const vector<int32> &seq_length){
         seq_device_ = seq_length_;
      }

   private:
      CuVectorG<int32> seq_device_;
};
#endif
