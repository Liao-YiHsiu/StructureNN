#include "my-cumatrix/cu-general.h"

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

// Instantiate classes
template class CuVectorG<int32>;
template class CuVectorG<uchar>;
template class CuVectorG<BaseFloat*>;

// ----------------------------------------------------------------------------------

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
