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

using namespace std;
using namespace kaldi;
using namespace kaldi::nnet1;

class CuIntVector;
template<typename Real> class myCuMatrix;
template<typename Real> class myCuVector;
template<class VALUE, class T> class ValueVectorPair;

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


typedef ValueVectorPair<BaseFloat, int32> ScorePath;
typedef SequentialTableReader<KaldiObjectHolder<ScorePath> > SequentialScorePathReader;
typedef TableWriter<KaldiObjectHolder<ScorePath> > ScorePathWriter;

double frame_acc(const vector<int32>& path1, const vector<int32>& path2);
double phone_acc(const vector<int32>& path1, const vector<int32>& path2);
int32 sample(const vector<BaseFloat> &arr);
int32 best(const vector<BaseFloat> &arr);
void trim_path(const vector<int32>& scr_path, vector<int32>& des_path);

void makeFeatureBatch(const Matrix<BaseFloat> &feat, const vector<int32> &path, int chgID, int32 maxState, SubMatrix<BaseFloat> mat);
void* makeFeatureP(void *param);
void makeFeature(const Matrix<BaseFloat> &feat, const vector<int32> &path, int32 maxState, SubVector<BaseFloat> vec);
void makeFeature(const CuMatrix<BaseFloat> &feat, const vector<int32> &path, int32 maxState, CuSubVector<BaseFloat> vec);

void makePost(double acc, Posterior &post);

bool updateLabelCuda(const myCuVector<BaseFloat> &arr, CuIntVector &lab, int S, BaseFloat &value);
void makeFeatureCuda(const myCuMatrix<BaseFloat> &feats, const CuIntVector &lab, int S, myCuMatrix<BaseFloat> &ret);

typedef struct{
   const Matrix<BaseFloat> *feat;
   vector<int32>           *path;
   int32                   maxState;
   SubMatrix<BaseFloat>    *mat;
   int32                   chgID;
} FData;


template<typename Real> class myCuMatrix : public CuMatrix<Real>{
   public:
      myCuMatrix():CuMatrix<Real>(){}
      myCuMatrix(CuMatrix<Real> &mat):CuMatrix<Real>(mat){}
      myCuMatrix(MatrixIndexT rows, MatrixIndexT cols, MatrixResizeType resize_type = kSetZero):
         CuMatrix<Real>(rows, cols, resize_type){}

      Real* Data(){return CuMatrix<Real>::Data();}
      const Real* Data()const {return CuMatrix<Real>::Data();}
};

template<typename Real> class myCuVector: public CuVector<Real>{
   public:
      myCuVector():CuVector<Real>(){}
      myCuVector(CuVector<Real> &vec):CuVector<Real>(vec){}
      myCuVector(MatrixIndexT dim, MatrixResizeType t = kSetZero):CuVector<Real>(dim, t){}
      Real* Data(){return CuVector<Real>::Data();}
      const Real* Data()const {return CuVector<Real>::Data();}
};

class CuIntVector{
   public:
      CuIntVector(int dim):data_(0), dim_(0){ Resize(dim); }
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
#endif
