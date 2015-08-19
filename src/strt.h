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
      StrtListBase(){}

      ~StrtListBase() { }

      void SetAll(int frames_N){frames_N_ = frames_N;}

      void Eval(const vector<BaseFloat> &nnet_target,
            const CuMatrixBase<BaseFloat> &nnet_out, CuMatrix<BaseFloat> *diff);

      string Report();

   private:

      double frames_;
      double correct_;
      double loss_;

      // partial results during training
      double frames_progress_;
      double correct_progress_;
      double loss_progress_;
      vector<float> loss_vec_;

      int frames_N_;

      Matrix<BaseFloat> nnet_out_host_;
      Matrix<BaseFloat> diff_host_;
};

#define NEW_STRT(name) \
   class name : public StrtBase{ \
      public: \
      name(bool pair, double error): StrtBase(pair, error) {}\
      protected: \
      void calcLoss(BaseFloat f_out, BaseFloat a_tgt, \
            BaseFloat &loss, BaseFloat &diff); \
} 

NEW_STRT(StrtMse);
NEW_STRT(StrtMgn);
NEW_STRT(StrtSoftmax);
NEW_STRT(StrtWSoftmax);
NEW_STRT(StrtExp);

#endif
