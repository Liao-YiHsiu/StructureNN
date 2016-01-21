#ifndef _MY_CU_MATRIX_H_
#define _MY_CU_MATRIX_H_

#include "cudamatrix/cu-matrix.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"

using namespace std;
using namespace kaldi;

// wrapper like CuMatrix with buffered mem
template<typename Real>
class MyCuMatrix : public CuMatrixBase<Real>{
   public:
      MyCuMatrix(): num_cols_r_(0), num_rows_r_(0) {}
      
      MyCuMatrix(MatrixIndexT rows, MatrixIndexT cols,
            MatrixResizeType resize_type = kSetZero): num_cols_r_(0), num_rows_r_(0){
         Resize(rows, cols, resize_type);
      }

      MyCuMatrix(const MyCuMatrix<Real> &other, 
            MatrixTransposeType trans = kNoTrans);

      MyCuMatrix<Real> &operator = (const CuMatrixBase<Real> &other){
         this->Resize(other.NumRows(), other.NumCols(), kUndefined);
         this->CopyFromMat(other);
         return *this;
      }

      MyCuMatrix<Real> &operator = (const CuMatrix<Real> &other){
         this->Resize(other.NumRows(), other.NumCols(), kUndefined);
         this->CopyFromMat(other);
         return *this;
      }

      MyCuMatrix<Real> &operator = (const MyCuMatrix<Real> &other){
         this->Resize(other.NumRows(), other.NumCols(), kUndefined);
         this->CopyFromMat(other);
         return *this;
      }

      MyCuMatrix<Real> &operator = (const MatrixBase<Real> &other){
         this->Resize(other.NumRows(), other.NumCols(), kUndefined);
         this->CopyFromMat(other);
         return *this;
      }

      void Resize(MatrixIndexT rows, MatrixIndexT cols,
            MatrixResizeType resize_type = kSetZero);

      ~MyCuMatrix() { Destroy(); }

   private:
      void Destroy();

      MatrixIndexT num_cols_r_;
      MatrixIndexT num_rows_r_;
};

#endif
