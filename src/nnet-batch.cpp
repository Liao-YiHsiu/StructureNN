#include "nnet-batch.h"

void BNnet::Propagate(const vector< CuMatrix<BaseFloat> > &in_arr, 
      vector< CuMatrix<BaseFloat> > &out_arr){

   CuMatrix<BaseFloat> forward;
   CuMatrix<BaseFloat> out;

   VecToMat(in_arr, forward);

   Nnet::Propagate(forward, &out);

   MatToVec(out, in_arr, out_arr);
}

void BNnet::Backpropagate(const vector< CuMatrix<BaseFloat> > &out_diff){

   CuMatrix<BaseFloat> backward;
   VecToMat(out_diff, backward);

   Nnet::Backpropagate(backward, NULL);
}

void BNnet::Feedforward(const vector< CuMatrix<BaseFloat> > &in_arr,
      vector< CuMatrix<BaseFloat> > &out_arr){

   CuMatrix<BaseFloat> forward;
   CuMatrix<BaseFloat> out;

   VecToMat(in_arr, forward);

   Nnet::Feedforward(forward, &out);

   MatToVec(out, in_arr, out_arr);
}

void BNnet::Feedforward(const CuMatrix<BaseFloat> &in, CuMatrix<BaseFloat> *out){
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
