#include "my-cumatrix/cu-matrix.h"

template<typename Real>
MyCuMatrix<Real>::MyCuMatrix(const MyCuMatrix<Real> &other,
      MatrixTransposeType trans): num_cols_r_(0), num_rows_r_(0) {
   if (trans == kNoTrans)
      this->Resize(other.NumRows(), other.NumCols(), kUndefined);
   else
      this->Resize(other.NumCols(), other.NumRows(), kUndefined);
   this->CopyFromMat(other, trans);
}

template<typename Real>
void MyCuMatrix<Real>::Resize(MatrixIndexT rows, MatrixIndexT cols,
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
void MyCuMatrix<Real>::Destroy(){
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

template class MyCuMatrix<float>;
template class MyCuMatrix<double>;
