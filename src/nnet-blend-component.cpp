#include "nnet-blend-component.h"

void Blend::Propagate(const CuMatrixBase<BaseFloat> &in,
      CuMatrix<BaseFloat> *out){
   assert( input_dim_ == in.NumCols() );

   resizeBuff(out, seq_length_.size(), output_dim_);
   CuSubMatrix<BaseFloat> sub = out->RowRange(0, seq_length_.size());
   PropagateFnc(in, &sub);
}

void Blend::Backpropagate(const CuMatrixBase<BaseFloat> &in,
      const CuMatrixBase<BaseFloat> &out,
      const CuMatrixBase<BaseFloat> &out_diff,
      CuMatrix<BaseFloat> *in_diff){
   assert( input_dim_ == in.NumCols() );
   assert( output_dim_ == out.NumCols() && out.NumRows() == seq_length_.size() );
   assert( out_diff.NumCols() == out.NumCols() && out_diff.NumRows() == out.NumRows());

   if(in_diff == NULL) return;

   resizeBuff(in_diff, in.NumRows(), in.NumCols());
   CuSubMatrix<BaseFloat> sub = in_diff->RowRange(0, in.NumRows());
   BackpropagateFnc(in, out, out_diff, &sub);
}


void Blend::SetSeqLengths(const vector<int32> &seq_length){
   seq_length_ = seq_length;

   SetSeqLengthsFnc(seq_length);
}

//---------------------------------------------------------------------------------------

Component* BlendSum::Copy() const{
   return new BlendSum(input_dim_, output_dim_);
}

void BlendSum::PropagateFnc(const CuMatrixBase<BaseFloat> &in,
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

void BlendSum::BackpropagateFnc(const CuMatrixBase<BaseFloat> &in,
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

void BlendSum::SetSeqLengthsFnc(const vector<int32> &seq_length){
   seq_device_ = seq_length_;
}
