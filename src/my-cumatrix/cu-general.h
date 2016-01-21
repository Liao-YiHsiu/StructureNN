#ifndef _MY_CU_GENERAL_H_
#define _MY_CU_GENERAL_H_

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "cudamatrix/cu-matrix.h"
#include "my-utils/type.h"
#include <vector>

using namespace std;
using namespace kaldi;

// add some buffer into it.
// it's a general purpose vector
template<typename T>
class CuVectorG{
   public:
      CuVectorG(int dim = 0):data_(0), dim_(0), dim_r_(0){ Resize(dim); }
      CuVectorG(const vector<T> &arr): data_(0), dim_(0){ Resize(arr.size()); CopyFromVec(arr); }
      CuVectorG(const CuVectorG &cuv);

      CuVectorG& operator=(const vector<T> &arr);

      ~CuVectorG(){ Destroy(); }

      void Destroy();

      int Dim() const{ return dim_; }

      T* Data(){ return data_; } 
      const T* Data()const{ return data_; }

      void CopyFromVec(const vector<T> &src);
      void CopyToVec(vector<T> &des)const;

      void Resize(int dim);

   private:
      T*   data_; 
      int  dim_;   // looked data dimension.
      int  dim_r_; // real data dimension.
};

// it's a general purpose matrix
template<typename T>
class CuMatrixG{
   public:
      CuMatrixG(int rows = 0, int cols = 0):rows_(0), cols_(0) { Resize(rows, cols); }
      CuMatrixG(const vector<T> &vec, int rows = 0, int cols = 0);
      CuMatrixG(const vector< vector<T> > &mat, int cols);

      ~CuMatrixG(){ Destroy(); }

      void Destroy();

      int NumRows() const { return rows_; }
      int NumCols() const { return cols_; }

      T* Data(){ return vec_.Data(); } 
      const T* Data()const{ return vec_.Data(); }

      void CopyFromVec(const vector<T> &src);
      void CopyFromVecVec(const vector< vector<T> > &src);
      void CopyToVec(vector<T> &des) const;
      void CopyToVecVec(vector< vector<T> > &dest) const;

      void Resize(int rows, int cols);

   private:
      int rows_;
      int cols_;
      CuVectorG<T> vec_;
};

#endif
