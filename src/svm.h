#include "nnet/nnet-trnopts.h"
#include "nnet/nnet-nnet.h"
#include "nnet/nnet-loss.h"
#include "nnet/nnet-randomizer.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "base/timer.h"
#include "cudamatrix/cu-device.h"
#include "util/edit-distance.h"
#include <sstream>

using namespace std;
using namespace kaldi;
using namespace kaldi::nnet1;

template<class VALUE, class T> class ValueVectorPair;
typedef ValueVectorPair<BaseFloat, int32> ScorePath;
typedef SequentialTableReader<KaldiObjectHolder<ScorePath> > SequentialScorePathReader;
typedef TableWriter<KaldiObjectHolder<ScorePath> > ScorePathWriter;

double frame_acc(const vector<int32>& path1, const vector<int32>& path2);
double phone_acc(const vector<int32>& path1, const vector<int32>& path2);
int32 sample(const vector<BaseFloat> &arr);
int32 best(const vector<BaseFloat> &arr);
void trim_path(const vector<int32>& scr_path, vector<int32>& des_path);

typedef struct{
   const Matrix<BaseFloat> *feat;
   vector<int32>           *path;
   int32                   maxState;
   SubMatrix<BaseFloat>    *mat;
   int32                   chgID;
} FData;

template<class V, class T> class ValueVectorPair{
   public:
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

double frame_acc(const vector<int32>& path1, const vector<int32>& path2){

   assert(path1.size() == path2.size());
   int corr = 0;
   for(int i = 0; i < path1.size(); ++i){
      corr += (path1[i] == path2[i]) ? 1:0;
   }

   return corr / (double)path1.size();
}

// reference is path1.
double phone_acc(const vector<int32>& path1, const vector<int32>& path2){
   assert(path1.size() == path2.size());

   vector<int32> path1_trim;
   vector<int32> path2_trim;
   trim_path(path1, path1_trim);
   trim_path(path2, path2_trim);

   int32 dist = LevenshteinEditDistance(path1_trim, path2_trim);

   int32 corr = path1_trim.size() - dist; 

   if(corr < 0) corr = 0;

   return corr / (double)path1_trim.size();
}



void makeFeature(const Matrix<BaseFloat> &feat, const vector<int32> &path, int32 maxState, SubVector<BaseFloat> vec){
   assert(feat.NumRows() == path.size());

   int feat_dim = feat.NumCols();

   SubVector<BaseFloat> tran(vec, feat_dim * maxState, maxState*maxState);
   for(int i = 0; i < path.size(); ++i){
      SubVector<BaseFloat> obs(vec, (path[i]-1)*feat_dim, feat_dim);
      //int32 offset = (path[i]-1)*feat_dim;
      //for(int k = 0; k < feat_dim; ++k)
      //   vec(offset+k) = feat(i,k); 
      obs.AddVec(1, feat.Row(i));

      if(i > 0){
         tran((path[i-1]-1)*maxState + path[i]-1) += 1;
      }
   }

   // normalization
   vec.Scale(1/(double)path.size());
}

void* makeFeatureP(void *param){
   FData* fData = (FData*) param;

   for(int i = 0; i < fData->maxState; ++i){
      vector<int32> path = *(fData->path);
      path[fData->chgID] = i+1;
      makeFeature(*(fData->feat), path, fData->maxState, fData->mat->Row(i));
   }
   return NULL;
}

void makeFeatureBatch(const Matrix<BaseFloat> &feat, const vector<int32> &path, int chgID, int32 maxState, SubMatrix<BaseFloat> mat){
   assert(feat.NumRows() == path.size());
   assert(mat.NumRows() == maxState);

   int feat_dim = feat.NumCols();

   // compute commonly used vector
   {
      SubVector<BaseFloat> vec = mat.Row(0);

      SubVector<BaseFloat> tran(vec, feat_dim * maxState, maxState*maxState);
      for(int i = 0; i < path.size(); ++i){
         if(i == chgID)continue;

         SubVector<BaseFloat> obs(vec, (path[i]-1)*feat_dim, feat_dim);
         obs.AddVec(1, feat.Row(i));

         if(i > 0 && i-1 != chgID){
            tran((path[i-1]-1)*maxState + path[i]-1) += 1;
         }
      }
      // copy to specified Matrix
      for(int i = 1; i < maxState; ++i)
         mat.Row(i).CopyFromVec(vec);
   }


   for(int i = 0; i < maxState; ++i){
      SubVector<BaseFloat> vec = mat.Row(i);

      SubVector<BaseFloat> tran(vec, feat_dim * maxState, maxState*maxState);
      SubVector<BaseFloat> obs(vec, i*feat_dim, feat_dim);
      obs.AddVec(1, feat.Row(chgID));

      if(chgID >= 1)
         tran((path[chgID-1]-1)*maxState + i) += 1;
      if(chgID+1 < path.size())
         tran(i*maxState + path[chgID+1]-1) += 1;
   }

   // normalization
   for(int i = 1; i < maxState; ++i)
         mat.Scale(1/(double)path.size());

}

void makeFeature(const CuMatrix<BaseFloat> &feat, const vector<int32> &path, int32 maxState, CuSubVector<BaseFloat> vec){
   assert(feat.NumRows() == path.size());

   int feat_dim = feat.NumCols();

   Vector<BaseFloat> tran_tmp(maxState*maxState);
   for(int i = 0; i < path.size(); ++i){
      CuSubVector<BaseFloat> obs(vec, (path[i]-1)*feat_dim, feat_dim);
      //int32 offset = (path[i]-1)*feat_dim;
      //for(int k = 0; k < feat_dim; ++k)
      //   vec(offset+k) = feat(i,k); 
      obs.AddVec(1, feat.Row(i));

      if(i > 0){
         tran_tmp((path[i-1]-1)*maxState + path[i]-1) += 1;
      }
   }

   CuSubVector<BaseFloat> tran(vec, feat_dim * maxState, maxState*maxState);
   tran.CopyFromVec(tran_tmp);

   // normalization
   vec.Scale(1/(double)path.size());
}

/*
void makePost(const vector<int32> &realPath, const vector<int32> &path, Posterior &post){
   double acc = path_acc(realPath, path);

   vector< pair<int32, BaseFloat> > arr; 

   if(acc != 0.0)
      arr.push_back(make_pair(0, acc));
   if(acc != 1.0)
      arr.push_back(make_pair(1, 1-acc));

   post.push_back(arr);
}
*/

void makePost(double acc, Posterior &post){
   vector< pair<int32, BaseFloat> > arr; 

   //if(acc != 0.0)
      arr.push_back(make_pair(0, acc));
   //if(acc != 1.0)
   //   arr.push_back(make_pair(1, 1-acc));

   post.push_back(arr);
}


int32 sample(const vector<BaseFloat> &arr){
   BaseFloat sum = 0;
   for(int i = 0; i < arr.size(); ++i)
      sum += arr[i];
   BaseFloat p = rand() / (double) RAND_MAX * sum;
   sum = 0;
   for(int i = 0; i < arr.size(); ++i){
      sum += arr[i];
      if(sum >= p ) return i;
   }
   assert(false);
   return -1;
}

int32 best(const vector<BaseFloat> &arr){
   assert(arr.size() >= 1);
   BaseFloat max = arr[0];
   int32 index = 0;
   for(int i = 1; i < arr.size(); ++i)
     if(max < arr[i]){
        max = arr[i];
        index = i;
     }
   return index;
}

void trim_path(const vector<int32>& scr_path, vector<int32>& des_path){
   des_path.clear();

   int32 prev = scr_path[0];
   des_path.push_back(scr_path[0]);

   for(int i = 1; i < scr_path.size(); ++i){
      if(prev != scr_path[i]){
         prev = scr_path[i];
         des_path.push_back(scr_path[i]);
      }
   }
}

