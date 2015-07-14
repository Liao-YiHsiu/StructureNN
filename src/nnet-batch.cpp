#include "nnet-batch.h"

void BNnet::Propagate(const vector< CuMatrix<BaseFloat> > &in_arr, 
      vector< CuMatrix<BaseFloat> > &out_arr){

   CuMatrix<BaseFloat> forward;
   CuMatrix<BaseFloat> out;

   VecToMat(in_arr, forward);

   Nnet::Propagate(forward, &out);

   MatToVec(out, in_arr, out_arr);
}

void BNnet::Backpropagate(const vector< CuMatrix<BaseFloat> > &out_diff,
      vector< CuMatrix<BaseFloat> > *in_diff){

   if(NumComponents() == 1 && GetComponent(0).GetType() == Component::kSplice){
      if(in_diff != NULL){
         if(in_diff->size() != out_diff.size())
            in_diff->resize(out_diff.size());
         for(int i = 0; i < out_diff.size(); ++i)
            (*in_diff)[i] = out_diff[i];
      }
      return;
   }

   CuMatrix<BaseFloat> backward;
   VecToMat(out_diff, backward);

   if( in_diff == NULL){
      Nnet::Backpropagate(backward, NULL);
   } else {
      CuMatrix<BaseFloat> backward_in_diff;
      Nnet::Backpropagate(backward, &backward_in_diff);
      MatToVec(backward_in_diff, out_diff, *in_diff);
   }
}

void BNnet::Feedforward(const vector< CuMatrix<BaseFloat> > &in_arr,
      vector< CuMatrix<BaseFloat> > &out_arr){

   CuMatrix<BaseFloat> forward;
   CuMatrix<BaseFloat> out;

   VecToMat(in_arr, forward);

   Nnet::Feedforward(forward, &out);

   MatToVec(out, in_arr, out_arr);
}

void BNnet::Propagate(const CuMatrixBase<BaseFloat> &in, CuMatrix<BaseFloat> *out) {
   Nnet::Propagate(in, out);
}

void BNnet::Backpropagate(const CuMatrixBase<BaseFloat> &out_diff, CuMatrix<BaseFloat> *in_diff){
   if(NumComponents() == 1 && GetComponent(0).GetType() == Component::kSplice){
      if(in_diff != NULL)
         *in_diff = out_diff;
      return;
   }
   Nnet::Backpropagate(out_diff, in_diff);
}

void BNnet::Feedforward(const CuMatrixBase<BaseFloat> &in, CuMatrix<BaseFloat> *out){
   Nnet::Feedforward(in, out);
}

void BNnet::VecToMat(const vector< CuMatrix<BaseFloat> > &arr, CuMatrix<BaseFloat> &mat){
   int Rows = 0;
   int Cols = arr[0].NumCols();

   for(int i = 0; i < arr.size(); ++i){
      KALDI_ASSERT( arr[i].NumCols() == Cols );
      Rows += arr[i].NumRows();
   }

   // copy each array element into a matrix
   mat.Resize(Rows, Cols, kUndefined);

   for(int i = 0, rowID = 0; i < arr.size(); ++i){
      int row = arr[i].NumRows();
      CuSubMatrix<BaseFloat> submatrix = mat.RowRange(rowID, row);
      submatrix.CopyFromMat(arr[i]);
      rowID += row;
   }
}

void BNnet::MatToVec(const CuMatrix<BaseFloat> &mat, const vector< CuMatrix<BaseFloat> > &ref,
      vector< CuMatrix<BaseFloat> > &arr){
   if(arr.size() != ref.size())
      arr.resize(ref.size());

   for(int i = 0, rowID = 0; i < ref.size(); ++i){
      int row = ref[i].NumRows();
      CuSubMatrix<BaseFloat> submatrix = mat.RowRange(rowID, row);
      arr[i] = submatrix;
      rowID += row;
   }
}
