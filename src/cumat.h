#ifndef _CUMAT_H_
#define _CUMAT_H_

#include "nnet/nnet-nnet.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "util.h"
#include <iostream>
#include <string>
#include <vector>

using namespace std;
using namespace kaldi;
using namespace kaldi::nnet1;

void VecToMat(const vector< CuMatrix<BaseFloat> > &arr, CuMatrix<BaseFloat> &mat, int N = -1);
void RepMat(const CuMatrix<BaseFloat> &src, CuMatrix<BaseFloat> &dest, int N = -1);

void MatToVec(const CuMatrix<BaseFloat> &mat, const vector< CuMatrix<BaseFloat> > &ref,
      vector< CuMatrix<BaseFloat> > &arr, int N = -1);
void MatToVec(const CuMatrix<BaseFloat> &mat, const vector<int> &ref,
      vector< CuMatrix<BaseFloat> > &arr);

vector<int> getRowsN(const vector< CuMatrix<BaseFloat> > &arr);

void Sum(const vector< CuMatrix<BaseFloat> > &arr, CuMatrix<BaseFloat>* out, int N = -1);

bool Same(const CuMatrixBase<BaseFloat> &a, const CuMatrixBase<BaseFloat> &b, double err = 1e-8);

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

// wrapper like CuMatrix with buffered mem
template<typename Real>
class myCuMatrix : public CuMatrixBase<Real>{
   public:
      myCuMatrix(): num_cols_r_(0), num_rows_r_(0) {}
      
      myCuMatrix(MatrixIndexT rows, MatrixIndexT cols,
            MatrixResizeType resize_type = kSetZero): num_cols_r_(0), num_rows_r_(0){
         Resize(rows, cols, resize_type);
      }

      myCuMatrix(const myCuMatrix<Real> &other, 
            MatrixTransposeType trans = kNoTrans);

      myCuMatrix<Real> &operator = (const CuMatrixBase<Real> &other){
         this->Resize(other.NumRows(), other.NumCols(), kUndefined);
         this->CopyFromMat(other);
         return *this;
      }

      myCuMatrix<Real> &operator = (const CuMatrix<Real> &other){
         this->Resize(other.NumRows(), other.NumCols(), kUndefined);
         this->CopyFromMat(other);
         return *this;
      }

      myCuMatrix<Real> &operator = (const MatrixBase<Real> &other){
         this->Resize(other.NumRows(), other.NumCols(), kUndefined);
         this->CopyFromMat(other);
         return *this;
      }

      void Resize(MatrixIndexT rows, MatrixIndexT cols,
            MatrixResizeType resize_type = kSetZero);

      ~myCuMatrix() { Destroy(); }

   private:
      void Destroy();

      MatrixIndexT num_cols_r_;
      MatrixIndexT num_rows_r_;
};

// ---------------------------------------------------------------------------------------------
// |                                  template class functions                                 |
// ---------------------------------------------------------------------------------------------

template<typename T>
CuVectorG<T>::CuVectorG(const CuVectorG &cuv){
   Resize(cuv.dim_);

   Timer tim;
   CU_SAFE_CALL(cudaMemcpy(data_, cuv.data_, dim_ * sizeof(T), cudaMemcpyDeviceToDevice));
   CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
   
}

template<typename T>
CuVectorG<T>& CuVectorG<T>::operator=(const vector<T> &src){
   Resize(src.size());
   CopyFromVec(src);
   return *this;
}


template<typename T>
void CuVectorG<T>::Destroy(){
   if(data_){
      CuDevice::Instantiate().Free(this->data_);
   }
   data_ = 0; dim_ = 0;
}

template<typename T>
void CuVectorG<T>::CopyFromVec(const vector<T> &src){
   if(src.size() != dim_)
      Resize(src.size());

   Timer tim;
   CU_SAFE_CALL(cudaMemcpy(data_, src.data(), dim_ * sizeof(T), cudaMemcpyHostToDevice));
   CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
}

template<typename T>
void CuVectorG<T>::CopyToVec(vector<T> &des)const{
   if(des.size() != dim_ ) des.resize(dim_);

   Timer tim;
   CU_SAFE_CALL(cudaMemcpy(des.data(), data_, dim_ * sizeof(T), cudaMemcpyDeviceToHost));
   CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
}

template<typename T>
void CuVectorG<T>::Resize(int dim){
   // don't set zeros
   if( dim <= dim_r_ ){
      dim_ = dim;
      return; 
   }

   Destroy();
   dim_ = dim;
   dim_r_ = dim;
   if(dim == 0) return;

   Timer tim;

   this->data_ = static_cast<T*>(CuDevice::Instantiate().Malloc(dim * sizeof(T)));

   CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());    
}

template<typename T>
CuMatrixG<T>::CuMatrixG(const vector<T> &vec, int rows, int cols) : rows_(0), cols_(0) {

   Resize(rows, cols); 
   CopyFromVec(vec);
}

template<typename T>
CuMatrixG<T>::CuMatrixG(const vector< vector<T> > &mat, int cols) : rows_(0), cols_(0) {

   Resize(mat.size(), cols); 
   CopyFromVecVec(mat);
}

template<typename T>
void CuMatrixG<T>::Destroy(){

   vec_.Destroy();
   rows_ = 0; cols_ = 0;
}

template<typename T>
void CuMatrixG<T>::CopyFromVec(const vector<T> &src){
   assert(src.size() == cols_ * rows_);

   Timer tim;
   CU_SAFE_CALL(cudaMemcpy(vec_.Data(), src.data(), cols_ * rows_ * sizeof(T), cudaMemcpyHostToDevice));
   CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
}

template<typename T>
void CuMatrixG<T>::CopyFromVecVec(const vector< vector<T> > &src){
   if(src.size() != rows_)
      Resize(src.size(), cols_);

   Timer tim;

   // check each cols 
   for(int i = 0; i < src.size(); ++i)
      assert(cols_ >= src[i].size());

   vector<T> tmp(rows_ * cols_);
   for(int i = 0; i < src.size(); ++i)
      for(int j = 0; j < src[i].size(); ++j){
         tmp[i * cols_ + j] = src[i][j];
      }
   CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());

   CopyFromVec(tmp);
}

template<typename T>
void CuMatrixG<T>::CopyToVec(vector<T> &des) const {
   if(des.size() != rows_ * cols_ ) des.resize(rows_ * cols_);

   Timer tim;
   CU_SAFE_CALL(cudaMemcpy(des.data(), vec_.Data(), rows_ * cols_ * sizeof(T), cudaMemcpyDeviceToHost));
   CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
}

template<typename T>
void CuMatrixG<T>::CopyToVecVec(vector< vector<T> > &dest) const {
   if(dest.size() != rows_){
      dest.resize(rows_);
      for(int i = 0; i < rows_; ++i)
         dest[i].resize(cols_);

   }else{
      for(int i = 0; i < dest.size(); ++i)
         assert(cols_ >= dest[i].size());
   }

   vector<T> tmp;
   CopyToVec(tmp);

   Timer tim;
   for(int i = 0; i < dest.size(); ++i)
      for(int j = 0; j < dest[i].size(); ++j){
         dest[i][j] = tmp[ i * cols_ + j];
      }
   CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
}

template<typename T>
void CuMatrixG<T>::Resize(int rows, int cols){
   Timer tim;
   vec_.Resize(rows * cols);

   rows_ = rows;
   cols_ = cols;
   CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
}

// ============================== myCuMatrix ==============================

template<typename Real>
myCuMatrix<Real>::myCuMatrix(const myCuMatrix<Real> &other,
      MatrixTransposeType trans): num_cols_r_(0), num_rows_r_(0) {
   if (trans == kNoTrans)
      this->Resize(other.NumRows(), other.NumCols(), kUndefined);
   else
      this->Resize(other.NumCols(), other.NumRows(), kUndefined);
   this->CopyFromMat(other, trans);
}

template<typename Real>
void myCuMatrix<Real>::Resize(MatrixIndexT rows, MatrixIndexT cols,
      MatrixResizeType resize_type){
   // This code does not currently support the other resize_type options.
   KALDI_ASSERT(resize_type == kSetZero || resize_type == kUndefined);
   if (rows * cols == 0) KALDI_ASSERT(rows == 0 && cols == 0);

   // buffered size...
   if (this->num_rows_r_ >= rows && this->num_cols_r_ >= cols) {
      this->num_rows_ = rows;
      this->num_cols_ = cols;
      if (resize_type == kSetZero) this->SetZero();
      return;
   }

   // resize ...
   if (this->num_rows_r_ != 0)
      this->Destroy();
#if HAVE_CUDA == 1
   if (CuDevice::Instantiate().Enabled()) {
      Timer tim;
      MatrixIndexT row_bytes = cols * sizeof(Real);
      size_t pitch;
      this->data_ = static_cast<Real*>(CuDevice::Instantiate().MallocPitch(
               row_bytes, rows, &pitch));
      this->num_rows_ = rows;
      this->num_cols_ = cols;
      this->num_rows_r_ = rows;
      this->num_cols_r_ = cols;
      this->stride_ = pitch / sizeof(Real);
      if (resize_type == kSetZero) this->SetZero();
      CuDevice::Instantiate().AccuProfile("CuMatrix::Resize", tim.Elapsed());
   }else
#endif
  { // Let the initializer of Matrix<Real> handle the allocation,
    // and then just do Swap which will switch the pointers.
    // This wastes a few instructions but is simple to code.
     MatrixIndexT skip;
     MatrixIndexT real_cols;
     size_t size;
     void *data;  // aligned memory block
     void *temp;  // memory block to be really freed

     // compute the size of skip and real cols
     skip = ((16 / sizeof(Real)) - cols % (16 / sizeof(Real)))
        % (16 / sizeof(Real));
     real_cols = cols + skip;
     size = static_cast<size_t>(rows) * static_cast<size_t>(real_cols)
        * sizeof(Real);

     // allocate the memory and set the right dimensions and parameters
     if (NULL != (data = KALDI_MEMALIGN(16, size, &temp))) {
        this->data_        = static_cast<Real *> (data);
        this->num_rows_      = rows;
        this->num_cols_      = cols;
        this->stride_  = real_cols;
     } else {
        throw std::bad_alloc();
     }
  }
   this->num_rows_r_ = this->num_rows_r_;
   this->num_cols_r_ = this->num_cols_r_;
}

template<typename Real>
void myCuMatrix<Real>::Destroy(){
#if HAVE_CUDA == 1
   if (CuDevice::Instantiate().Enabled()) {
      if (this->data_ != NULL) {
         Timer tim;
         CuDevice::Instantiate().Free(this->data_);
         CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
      }
   } else 
#endif
   {
      if (this->data_ != NULL) KALDI_MEMALIGN_FREE(this->data_);
   }
   this->data_ = NULL;
   this->num_rows_ = 0;
   this->num_cols_ = 0;
   this->num_rows_r_ = 0;
   this->num_cols_r_ = 0;
   this->stride_ = 0;
}

#endif
