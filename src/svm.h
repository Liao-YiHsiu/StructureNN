#include "nnet/nnet-trnopts.h"
#include "nnet/nnet-nnet.h"
#include "nnet/nnet-loss.h"
#include "nnet/nnet-randomizer.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "base/timer.h"
#include "cudamatrix/cu-device.h"
#include <sstream>

using namespace std;
using namespace kaldi;
using namespace kaldi::nnet1;

template<class VALUE, class T> class ValueVectorPair;
typedef ValueVectorPair<BaseFloat, int32> ScorePath;

double path_acc(const vector<int32> path1, const vector<int32> path2);

template<class V, class T> class ValueVectorPair{
   public:
      typedef vector< pair < V, vector<T> > > Table;

      ValueVectorPair(){}
      ValueVectorPair(const Table& value): val(value){}

      void Write(ostream &os, bool binary) const{
         int32 vecsz = static_cast<int32>(val.size());
         KALDI_ASSERT((size_t)vecsz == val.size());

         if(binary){
            os.write(reinterpret_cast<const char *>(&vecsz), sizeof(vecsz));
         }else{
            os << vecsz << endl ;
         }

         for(int i = 0; i < val.size(); ++i){
            WriteBasicType(os, binary, val[i].first);
            WriteIntegerVector(os, binary, val[i].second);
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

double path_acc(const vector<int32> path1, const vector<int32> path2){

   assert(path1.size() == path2.size());
   int corr = 0;
   for(int i = 0; i < path1.size(); ++i){
      corr += (path1[i] == path2[i]) ? 1:0;
   }

   return corr / (double)path1.size();
}

void makeFeature(const Matrix<BaseFloat> &feat, const vector<int32> &path, int32 maxState, SubVector<BaseFloat> vec){
   assert(feat.NumRows() == path.size());

   int feat_dim = feat.NumCols();

   SubVector<BaseFloat> tran(vec, feat_dim * maxState, maxState*maxState);
   for(int i = 0; i < path.size(); ++i){
      SubVector<BaseFloat> obs(vec, (path[i]-1)*feat_dim, feat_dim);
      obs.CopyFromVec(feat.Row(i));

      if(i > 0){
         tran((path[i-1]-1)*maxState + path[i]-1) += 1;
      }
   }

   // normalization
   vec.Scale(1/(double)path.size());
}

void makePost(const vector<int32> &realPath, const vector<int32> &path, Posterior &post){
   double acc = path_acc(realPath, path);

   vector< pair<int32, BaseFloat> > arr; 

   if(acc != 0.0)
      arr.push_back(make_pair(0, acc));
   if(acc != 1.0)
      arr.push_back(make_pair(1, 1-acc));

   post.push_back(arr);
}


