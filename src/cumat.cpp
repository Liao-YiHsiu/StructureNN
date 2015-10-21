#include "cumat.h"

void VecToMat(const vector< CuMatrix<BaseFloat> > &arr, CuMatrix<BaseFloat> &mat, int N){
   int Rows = 0;
   int Cols = arr[0].NumCols();
   if(N < 0) N = arr.size();

   for(int i = 0; i < N; ++i){
      KALDI_ASSERT( arr[i].NumCols() == Cols );
      Rows += arr[i].NumRows();
   }

   // copy each array element into a matrix
   mat.Resize(Rows, Cols, kUndefined);

   for(int i = 0, rowID = 0; i < N; ++i){
      int row = arr[i].NumRows();
      CuSubMatrix<BaseFloat> submatrix = mat.RowRange(rowID, row);
      submatrix.CopyFromMat(arr[i]);
      rowID += row;
   }
}

void MatToVec(const CuMatrix<BaseFloat> &mat, const vector< CuMatrix<BaseFloat> > &ref,
      vector< CuMatrix<BaseFloat> > &arr, int N){
   if( N < 0) N = ref.size();
   vector<int> ref_vec = getRowsN(ref);
   ref_vec.resize(N);

   MatToVec(mat, ref_vec, arr);
}

void MatToVec(const CuMatrix<BaseFloat> &mat, const vector<int> &ref,
      vector< CuMatrix<BaseFloat> > &arr){
   if(arr.size() != ref.size())
      arr.resize(ref.size());

   int rowID = 0;
   for(int i = 0; i < ref.size(); ++i){
      int row = ref[i];
      CuSubMatrix<BaseFloat> submatrix = mat.RowRange(rowID, row);
      arr[i] = submatrix;
      rowID += row;
   }
   assert(rowID == mat.NumRows());
}

void RepMat(const CuMatrix<BaseFloat> &src, CuMatrix<BaseFloat> &dest, int N){
   assert(N > 0);
   int row = src.NumRows();
   int Rows = row * N;
   int Cols = src.NumCols();

   dest.Resize(Rows, Cols, kUndefined);

   for(int i = 0; i < N; ++i){
      CuSubMatrix<BaseFloat> submatrix = dest.RowRange(i*row, row);
      submatrix.CopyFromMat(src);
   }
}

vector<int> getRowsN(const vector< CuMatrix<BaseFloat> > &arr){
   vector<int> ret(arr.size());
   for(int i = 0; i < arr.size(); ++i)
      ret[i] = arr[i].NumRows();

   return ret;
}

void Sum(const vector< CuMatrix<BaseFloat> > &arr, CuMatrix<BaseFloat>* out, int N){
   if(N < 0) N = arr.size();
   *out = arr[0];
   for(int i = 1; i < N; ++i)
      out->AddMat(1.0, arr[i]);
}

bool Same(const CuMatrixBase<BaseFloat> &a, const CuMatrixBase<BaseFloat> &b, double err){
   CuMatrix<BaseFloat> c(a.NumRows(), a.NumCols(), kUndefined);
   c = a;
   c.AddMat(-1, b);
   c.ApplyPow(2);
   return c.Sum() < err;
}
