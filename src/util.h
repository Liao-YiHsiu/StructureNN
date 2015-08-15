#ifndef _SVM_H
#define _SVM_H
#include "nnet/nnet-trnopts.h"
#include "nnet/nnet-nnet.h"
#include "nnet/nnet-loss.h"
#include "nnet/nnet-randomizer.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "base/timer.h"
#include "cudamatrix/cu-device.h"
#include "util/edit-distance.h"
#include "kernel.h"
#include <sstream>
#include <iterator>
#include <map>
#include <string>
#include <math.h>
#include <float.h>

#define BUFSIZE 4096

using namespace std;
using namespace kaldi;
using namespace kaldi::nnet1;

class CuIntVector;
template<typename Real> class myCuMatrix;
template<class VALUE, class T> class ValueVectorPair;

typedef unsigned char uchar;

template<class V, class T> class ValueVectorPair{
   public:
      friend class ValueVectorPair<float, int>;
      typedef vector< pair < V, vector<T> > > Table;
      ValueVectorPair(){}
      ValueVectorPair(const Table& value): val(value){}
      void Write(ostream &os, bool binary) const{
         try{
            int32 vecsz = static_cast<int32>(val.size());
            KALDI_ASSERT((size_t)vecsz == val.size());

            if(binary){
               os.write(reinterpret_cast<const char *>(&vecsz), sizeof(vecsz));
            }else{
               os << vecsz << " " ;
            }

            for(int i = 0; i < val.size(); ++i){
               WriteBasicType(os, binary, val[i].first);
               WriteIntegerVector(os, binary, val[i].second);
            }
         } catch(const std::exception &e) {
            std::cerr << e.what();
            exit(-1);
         }
      }

      void Read(istream &is, bool binary){
         int32 vecsz;
         if(binary){
            is.read(reinterpret_cast<char *>(&vecsz), sizeof(vecsz));
         }else{
            is >> vecsz >> std::ws;
         }

         val.resize(vecsz);
         for(int i = 0; i < vecsz; ++i){
            ReadBasicType(is, binary, &val[i].first);
            ReadIntegerVector(is, binary, &val[i].second);
         }
      }
      const Table& Value() const{ return val; }
      Table& Value(){ return val; }

   private:
      Table val;
};

typedef ValueVectorPair<BaseFloat, uchar> ScorePath;

typedef SequentialTableReader<KaldiObjectHolder<ScorePath> > SequentialScorePathReader;
typedef TableWriter<KaldiObjectHolder<ScorePath> >           ScorePathWriter;

typedef SequentialTableReader<BasicVectorHolder<uchar> >     SequentialUcharVectorReader;
typedef RandomAccessTableReader<BasicVectorHolder<uchar> >   RandomAccessUcharVectorReader;
typedef TableWriter<BasicVectorHolder<uchar> >               UcharVectorWriter;

inline double sigmoid(double x);

double frame_acc(const vector<uchar>& path1, const vector<uchar>& path2, double param = 1.0);
double phone_acc(const vector<uchar>& path1, const vector<uchar>& path2, double param = 1.0);
int32 best(const vector<BaseFloat> &arr);
void trim_path(const vector<uchar>& scr_path, vector<uchar>& des_path);
void UcharToInt32(const vector<uchar>& src_path, vector<int32>& des_path);
void Int32ToUchar(const vector<int32>& src_path, vector<uchar>& des_path);

void getPhone(const string &key, const string &timit, map<string, int> &phMap, vector<uchar> &phIdx);
void readPhMap(const string path, const string id_path, map<string, int> &phMap);
string execute(const string &cmd);

void print(const MatrixBase<BaseFloat> &mat, int row = -1);
void print(const CuMatrixBase<BaseFloat> &mat, int row = -1);

// get GPU memory pointer in a dirty way.
template<typename Real>
Real* getCuPointer(CuMatrixBase<Real> *matrix){
   myCuMatrix<Real>* mat_ptr = (myCuMatrix<Real>*)(void*)matrix;
   return mat_ptr->Data();
}

void propPsi(int N, int F, int S, int maxL, PsiPack* packs_ptr);
void backPsi(int N, int F, int S, int maxL, PsiPack* packs_ptr);

typedef struct{
   const Matrix<BaseFloat> *feat;
   vector<uchar>           *path;
   int32                   maxState;
   SubMatrix<BaseFloat>    *mat;
   int32                   chgID;
} FData;


template<typename Real>
class myCuMatrix : public CuMatrixBase<Real>{
   public:
      Real* Data(){return CuMatrixBase<Real>::Data();}
      const Real* Data()const {return CuMatrixBase<Real>::Data();}
};

// it's a general purpose vector
template<typename T>
class CuVectorG{
   public:
      CuVectorG(int dim = 0):data_(0), dim_(0){ Resize(dim); }
      CuVectorG(const vector<T> &arr): data_(0), dim_(0){ Resize(arr.size()); CopyFromVec(arr); }

      ~CuVectorG(){ Destroy(); }

      void Destroy(){
         if(data_){
            CuDevice::Instantiate().Free(this->data_);
         }
         data_ = 0; dim_ = 0;
      }

      int Dim() const{ return dim_; }
      T*  Data(){ return data_; } 
      const T* Data()const{ return data_; }

      void CopyFromVec(const vector<T> &src){
         if(src.size() != dim_)
            Resize(src.size());

         Timer tim;
         CU_SAFE_CALL(cudaMemcpy(data_, src.data(), dim_ * sizeof(T), cudaMemcpyHostToDevice));
         CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
      }

      void CopyToVec(vector<T> &des)const{
         if(des.size() != dim_ ) des.resize(dim_);

         Timer tim;
         CU_SAFE_CALL(cudaMemcpy(des.data(), data_, dim_ * sizeof(T), cudaMemcpyDeviceToHost));
         CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
      }

      void Resize(int dim){
         if( dim_ == dim ) return; // don't set zeros

         Destroy();
         dim_ = dim;
         if(dim == 0) return;

         Timer tim;

         this->data_ = static_cast<T*>(CuDevice::Instantiate().Malloc(dim * sizeof(T)));

         CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());    
      }

   private:
      T*   data_; 
      int  dim_;
};

// it's a general purpose matrix
template<typename T>
class CuMatrixG{
   public:
      CuMatrixG(int rows = 0, int cols = 0):rows_(0), cols_(0){ Resize(rows, cols); }
      CuMatrixG(const vector<T> &vec, int rows = 0, int cols = 0):rows_(0), cols_(0)
   { Resize(rows, cols); CopyFromVec(vec); }
      CuMatrixG(const vector< vector<T> > &mat, int cols): rows_(0), cols_(0)
   { Resize(mat.size(), cols); CopyFromVecVec(mat); }

      ~CuMatrixG(){ Destroy(); }

      void Destroy(){
         vec_.Destroy();
         rows_ = 0; cols_ = 0;
      }

      int NumRows() const { return rows_; }
      int NumCols() const { return cols_; }

      T* Data(){ return vec_.Data(); } 
      const T* Data()const{ return vec_.Data(); }

      void CopyFromVec(const vector<T> &src){
         assert(src.size() == cols_ * rows_);

         Timer tim;
         CU_SAFE_CALL(cudaMemcpy(vec_.Data(), src.data(), cols_ * rows_ * sizeof(T), cudaMemcpyHostToDevice));
         CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
      }

      void CopyFromVecVec(const vector< vector<T> > &src){
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

      void CopyToVec(vector<T> &des) const {
         if(des.size() != rows_ * cols_ ) des.resize(rows_ * cols_);

         Timer tim;
         CU_SAFE_CALL(cudaMemcpy(des.data(), vec_.Data(), rows_ * cols_ * sizeof(T), cudaMemcpyDeviceToHost));
         CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
      }

      void CopyToVecVec(vector< vector<T> > &dest) const {
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

      void Resize(int rows, int cols){
         Timer tim;
         vec_.Resize(rows * cols);

         rows_ = rows;
         cols_ = cols;
         CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
      }

   private:
      int rows_;
      int cols_;
      CuVectorG<T> vec_;
};

class CuIntVector{
   public:
      CuIntVector(int dim = 0):data_(0), dim_(0){ Resize(dim); }
      CuIntVector(const vector<int> &arr): data_(0), dim_(0){ Resize(arr.size()); CopyFromVec(arr); }

      ~CuIntVector(){ Destroy(); }

      void Destroy();

      int Dim() const{ return dim_; }
      int* Data(){ return data_; } 
      const int* Data()const{ return data_; }

      void CopyFromVec(const vector<int> &src);
      void CopyToVec(vector<int> &des)const;

      void Resize(int dim);

   private:
      int* data_; 
      int  dim_;
};

enum LABEL { ALL_TYPE = 0, REF_TYPE, LAT_TYPE, RAND_TYPE, END_TYPE };
const string LABEL_NAME[] = {"ALL_TYPE", "REF_TYPE", "LAT_TYPE", "RAND_TYPE", "END_TYPE"};

// Structure learning loss
class Strt {
   public:
      Strt(): frames_arr_(END_TYPE), correct_arr_(END_TYPE), loss_arr_(END_TYPE),
      frames_progress_(0), correct_progress_(0), loss_progress_(0), diff_host_(2) {}

      ~Strt() { }

      /// Evaluate cross entropy using target-matrix (supports soft labels),
      /// nnet_out = (raw >= 0) ? f(x,y) : f(x, y) - f(x, y_hat)
      /// counter  = # of errors
      /// returns the index of max f(x,y) + delta(y, y_hat);
      int Eval(const VectorBase<BaseFloat> &delta, const CuMatrixBase<BaseFloat> &nnet_out, 
            vector<CuMatrix<BaseFloat> > *diff, int* counter = NULL, int raw= -1, 
            const vector<int>* example_type = NULL);

      string Report();

   private: 
      vector<double> frames_arr_;
      vector<double> correct_arr_;
      vector<double> loss_arr_;

      // partial results during training
      double frames_progress_;
      double correct_progress_;
      double loss_progress_;
      vector<float> loss_vec_;

      Matrix<BaseFloat>      nnet_out_host_;
      vector<Matrix<BaseFloat> > diff_host_;
};

// Structure learning loss using comparision between two label seqs.
class StrtCmp{
   public:
      StrtCmp(): frames_arr_(END_TYPE*END_TYPE), correct_arr_(END_TYPE*END_TYPE), loss_arr_(END_TYPE*END_TYPE),
      frames_progress_(0), correct_progress_(0), loss_progress_(0),
      frames_N_(0), diff_host_(2) {}

      ~StrtCmp() { }

      void SetAll(int frames_N){frames_N_ = frames_N;}

      /// Evaluate cross entropy using target-matrix (supports soft labels),
      /// nnet_out = (raw >= 0) ? f(x,y) : f(x, y) - f(x, y_hat)
      void Eval(const VectorBase<BaseFloat> &delta, const CuMatrixBase<BaseFloat> &nnet_out, 
            vector<CuMatrix<BaseFloat> > *diff, const vector<int>* example_type = NULL);

      string Report();

      virtual void calcErr(BaseFloat f_diff, BaseFloat a_diff,
            BaseFloat &error, BaseFloat &diff1, BaseFloat &diff2) = 0;

   pritvate:
      vector<double> frames_arr_;
      vector<double> correct_arr_;
      vector<double> loss_arr_;

      // partial results during training
      double frames_progress_;
      double correct_progress_;
      double loss_progress_;
      vector<float> loss_vec_;

      int frames_N_;

      Matrix<BaseFloat>      nnet_out_host_;
      vector<Matrix<BaseFloat> > diff_host_;
};

class StrtBase{
   public:
      StrtBase(int types_num = END_TYPE):
         frames_arr_(types_num), correct_arr_(types_num), loss_arr_(types_num),
         frames_progress_(0), correct_progress_(0), loss_progress_(0),
         frames_N_(0), diff_host_(2) {}

      ~StrtBase() { }

      void SetAll(int frames_N){frames_N_ = frames_N;}

      /// Evaluate cross entropy using target-matrix (supports soft labels),
      /// nnet_out = (raw >= 0) ? f(x,y) : f(x, y) - f(x, y_hat)
      void Eval(const VectorBase<BaseFloat> &delta, const CuMatrixBase<BaseFloat> &nnet_out, 
            vector<CuMatrix<BaseFloat> > *diff, const vector<int>* example_type = NULL,
            int bias= -1, int* counter = NULL, int *maxErrId = NULL );

      string Report();

      virtual void calcErr(BaseFloat f_diff, BaseFloat a_diff,
            BaseFloat &error, BaseFloat &diff1, BaseFloat &diff2) = 0;

   pritvate:
      vector<double> frames_arr_;
      vector<double> correct_arr_;
      vector<double> loss_arr_;

      // partial results during training
      double frames_progress_;
      double correct_progress_;
      double loss_progress_;
      vector<float> loss_vec_;

      int frames_N_;

      Matrix<BaseFloat>      nnet_out_host_;
      vector<Matrix<BaseFloat> > diff_host_;
}
#endif
