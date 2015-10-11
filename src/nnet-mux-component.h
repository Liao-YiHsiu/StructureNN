
#ifndef MUX_COMPONENT_H_
#define MUX_COMPONENT_H_

#include "nnet/nnet-component.h"
#include "nnet/nnet-utils.h"
#include "cudamatrix/cu-math.h"
#include "util/text-utils.h"
#include "util.h"

#include <algorithm>
#include <sstream>

using namespace std;

//class Mux: public UpdatableComponent {
class Mux {
   public:
      Mux(int32 dim_in, int32 dim_out, int size)
         : input_dim_(dim_in), output_dim_(dim_out), comps_(size),
         in_buff_(size), in_diff_buff_(size), out_buff_(size), out_diff_buff_(size),
         cnt_(size), seq_stride_(1) { }
      ~Mux(){ Destroy(); }
      
      void Destroy();

      int32 InputDim() const{ return input_dim_; }
      int32 OutputDim() const{ return output_dim_; }

      // seq is 0-based
      // each input row could have multiple seqs.
      void setSeqs(const vector<int32> &seq, int seq_stride);

      void Propagate(const CuMatrixBase<BaseFloat> &in, CuMatrix<BaseFloat> *out);
      void Backpropagate(const CuMatrixBase<BaseFloat> &in,
            const CuMatrixBase<BaseFloat> &out,
            const CuMatrixBase<BaseFloat> &out_diff,
            CuMatrix<BaseFloat> *in_diff);

      static Mux* Init(istream &is);
      static Mux* Read(istream &is, bool binary);
      void Write(ostream &os, bool binary) const;

      string Info() const;
      string InfoGradient() const;

      Mux* Copy() const;

      int NumComponents() const { return comps_.size(); }

      // updatable componets
      int32 NumParams() const;
      void Update(const CuMatrixBase<BaseFloat> &input, const CuMatrixBase<BaseFloat> &diff);
      void SetTrainOptions(const NnetTrainOptions &opts);
      const NnetTrainOptions& GetTrainOptions() const{ return opts_; }

   private:
      BaseFloat** getVecCuMatrixPt(vector< CuMatrix<BaseFloat> > &mat_arr);

      int32  input_dim_, output_dim_;

      vector< Component* > comps_;
      vector< CuMatrix<BaseFloat> > in_buff_; 
      vector< CuMatrix<BaseFloat> > in_diff_buff_; 
      vector< CuMatrix<BaseFloat> > out_buff_;
      vector< CuMatrix<BaseFloat> > out_diff_buff_;
      vector<int32> seq_;
      vector<int32> id_;
      vector<int32> cnt_;

      CuVectorG<int32> seq_device_;
      CuVectorG<int32> id_device_;

      CuVectorG< BaseFloat* > mat_arr_pt_device_;

      int seq_stride_;

      NnetTrainOptions opts_;
};

#endif
