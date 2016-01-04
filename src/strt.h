#ifndef _STRT_H_
#define _STRT_H_

#include "nnet/nnet-nnet.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "util.h"
#include <iostream>
#include <string>
#include <vector>
#include <iterator>
#include <sstream>
#include <algorithm>

using namespace std;
using namespace kaldi;
using namespace kaldi::nnet1;

enum LABEL { ALL_TYPE = 0, REF_TYPE, LAT_TYPE, RAND_TYPE, END_TYPE };
const string LABEL_NAME[] = {"ALL_TYPE", "REF_TYPE", "LAT_TYPE", "RAND_TYPE", "END_TYPE"};

class StrtBase{
   public:
      StrtBase(bool pair, double error);

      ~StrtBase() { }

      void SetAll(int frames_N){frames_N_ = frames_N;}

      void Eval(const VectorBase<BaseFloat> &delta, const CuMatrixBase<BaseFloat> &nnet_out, 
            vector<CuMatrix<BaseFloat> > *diff_dev, const vector<int>* example_type = NULL);

      string Report();

      static StrtBase* getInstance(string name, bool pair, double error);

   protected:
      virtual void calcLoss(BaseFloat f_out, BaseFloat a_tgt,
            BaseFloat &loss, BaseFloat &diff) = 0;

   private:
      string getStr(int index);

      bool pair_;
      double error_;

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

class StrtListBase{
   public:
      StrtListBase(double sigma, double error):
         sigma_(sigma), error_(error), T_(10),
         frames_(0), correct_(0), loss_(0), ndcg_(0),
         frames_progress_(0), correct_progress_(0), loss_progress_(0), ndcg_progress_(0),
         frames_N_(0) {}

      ~StrtListBase() { }

      void SetAll(int frames_N){frames_N_ = frames_N;}

      void Eval(const vector<BaseFloat> &nnet_target,
            const CuMatrixBase<BaseFloat> &nnet_out, CuMatrix<BaseFloat> *diff);

      void Eval(const vector<BaseFloat> &nnet_target,
            const MatrixBase<BaseFloat> &nnet_out, Matrix<BaseFloat> *diff);

      string Report();

      static StrtListBase* getInstance(string name, double sigma = 1.0, double error = 0);

   protected:
      virtual void calcLoss(const vector<BaseFloat> &nnet_target, 
            const vector<int> &index_t, const vector<int> &index_f,
            const vector<BaseFloat> &relevance, BaseFloat &loss,
            const Matrix<BaseFloat> &nnet_out_host, Matrix<BaseFloat> *diff_host
            ) = 0;


      double sigma_;
      double error_;
      int T_;

   private:

      double frames_;
      double correct_;
      double loss_;
      double ndcg_;

      // partial results during training
      double frames_progress_;
      double correct_progress_;
      double loss_progress_;
      double ndcg_progress_;
      vector<float> loss_vec_;

      int frames_N_;


};

class StrtBest{
   public:
      StrtBest(double sigma):
         sigma_(sigma),
         frames_(0), correct_(0), loss_(0),
         frames_progress_(0), correct_progress_(0), loss_progress_(0),
         frames_N_(0) {}

      ~StrtBest() { }

      void SetAll(int frames_N){frames_N_ = frames_N;}

      void Eval(int bestIdx, const CuMatrixBase<BaseFloat> &nnet_out, CuMatrix<BaseFloat> *diff);

      string Report();

   protected:

      Matrix<BaseFloat> nnet_out_host_;
      Matrix<BaseFloat> diff_host_;

      double sigma_;

   private:

      double frames_;
      double correct_;
      double loss_;

      // partial results during training
      double frames_progress_;
      double correct_progress_;
      double loss_progress_;

      vector<float> loss_vec_;
      vector<double> tmp_vec_;

      int frames_N_;
};

#define NEW_STRT_PAIR(name) \
   class name : public StrtBase{ \
      public: \
      name(bool pair, double error): StrtBase(pair, error) {}\
      protected: \
      void calcLoss(BaseFloat f_out, BaseFloat a_tgt, \
            BaseFloat &loss, BaseFloat &diff); \
} 

NEW_STRT_PAIR(StrtMse);
NEW_STRT_PAIR(StrtMgn);
NEW_STRT_PAIR(StrtSoftmax);
NEW_STRT_PAIR(StrtWSoftmax);
NEW_STRT_PAIR(StrtExp);

#define NEW_STRT_LIST(name) \
   class name : public StrtListBase{ \
      public: \
      name(double sigma, double error): StrtListBase(sigma, error) {}\
      protected: \
      void calcLoss(const vector<BaseFloat> &nnet_target, \
            const vector<int> &index_t, const vector<int> &index_f, \
            const vector<BaseFloat> &relevance, BaseFloat &loss, \
            const Matrix<BaseFloat> &nnet_out_host, Matrix<BaseFloat> *diff_host \
            ); \
} 
NEW_STRT_LIST(StrtListNet);
NEW_STRT_LIST(StrtListRelu);
NEW_STRT_LIST(StrtRankNet);
NEW_STRT_LIST(StrtLambdaRank);

#endif
