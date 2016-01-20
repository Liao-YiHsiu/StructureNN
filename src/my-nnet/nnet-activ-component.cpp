#include "nnet-activ-component.h"

Component* ReLU::Copy() const { 
   return new ReLU(input_dim_, output_dim_);
}

void ReLU::PropagateFnc(const CuMatrixBase<BaseFloat> &in,
      CuMatrixBase<BaseFloat> *out){
   out->CopyFromMat(in);
   out->ApplyFloor(0.0);
}

void ReLU::BackpropagateFnc(const CuMatrixBase<BaseFloat> &in,
      const CuMatrixBase<BaseFloat> &out,
      const CuMatrixBase<BaseFloat> &out_diff,
      CuMatrixBase<BaseFloat> *in_diff){

   in_diff->CopyFromMat(out);
   in_diff->ApplyHeaviside();
   in_diff->MulElements(out_diff);
}
