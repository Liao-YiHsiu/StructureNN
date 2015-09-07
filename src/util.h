#ifndef _UTIL_H
#define _UTIL_H
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
#include "cumat.h"
#include "strt.h"
#include <sstream>
#include <iterator>
#include <map>
#include <string>
#include <math.h>
#include <float.h>
#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/file.h>
#include <stdlib.h>

#define BUFSIZE 4096
#define GPU_FILE "/tmp/gpu_lock"

using namespace std;
using namespace kaldi;
using namespace kaldi::nnet1;

typedef unsigned char uchar;
template<class VALUE, class T> class ValueVectorPair;

typedef ValueVectorPair<BaseFloat, uchar> ScorePath;

typedef SequentialTableReader<KaldiObjectHolder<ScorePath> > SequentialScorePathReader;
typedef TableWriter<KaldiObjectHolder<ScorePath> >           ScorePathWriter;

typedef SequentialTableReader<BasicVectorHolder<uchar> >     SequentialUcharVectorReader;
typedef RandomAccessTableReader<BasicVectorHolder<uchar> >   RandomAccessUcharVectorReader;
typedef TableWriter<BasicVectorHolder<uchar> >               UcharVectorWriter;

inline double sigmoid(double x){ return 1/(1+exp(-x)); }
inline double softplus(double x){ return x > 0 ? x + softplus(-x) : log(1+exp(x)); }
inline double log_add(double a, double b){ return a + softplus(b-a);}

double frame_acc(const vector<uchar>& path1, const vector<uchar>& path2, bool norm = true);
double phone_acc(const vector<uchar>& path1, const vector<uchar>& path2, bool norm = true);

int32 best(const vector<BaseFloat> &arr);

void trim_path(const vector<uchar>& scr_path, vector<uchar>& des_path);
void UcharToInt32(const vector<uchar>& src_path, vector<int32>& des_path);
void Int32ToUchar(const vector<int32>& src_path, vector<uchar>& des_path);


void getPhone(const string &key, const string &timit, map<string, int> &phMap, vector<uchar> &phIdx);
void readPhMap(const string path, const string id_path, map<string, int> &phMap);
string execute(const string &cmd);

string strAfter(const string &src, const string &key);

void print(const MatrixBase<BaseFloat> &mat, int row = -1);
void print(const CuMatrixBase<BaseFloat> &mat, int row = -1);


void propPsi(int N, int F, int S, int maxL, PsiPack* packs_ptr);
void backPsi(int N, int F, int S, int maxL, PsiPack* packs_ptr);

void propRPsi(RPsiPack *pack);
void backRPsi(RPsiPack *pack);

void LockSleep(string filename, int ms = 2000);

template<typename T>
void VecToVecRef(vector<T>& src, vector<T*> &dest);

template<class V, class T> class ValueVectorPair{
   public:
      friend class ValueVectorPair<float, int>;
      typedef vector< pair < V, vector<T> > > Table;

      ValueVectorPair(){}
      ValueVectorPair(const Table& value): val(value){}

      void Write(ostream &os, bool binary) const;
      void Read(istream &is, bool binary);

      const Table& Value() const{ return val; }
      Table& Value(){ return val; }

   private:
      Table val;
};

// ---------------------------------------------------------------------------------------------
// |                                  template class functions                                 |
// ---------------------------------------------------------------------------------------------


template<class V, class T>
void ValueVectorPair<V, T>::Write(ostream &os, bool binary) const{
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

template<class V, class T>
void ValueVectorPair<V, T>::Read(istream &is, bool binary){
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

template<typename T>
void VecToVecRef(vector<T>& src, vector<T*> &dest){
   dest.resize(src.size());
   for(int i = 0; i < src.size(); ++i)
      dest[i] = &src[i];
}

#endif
