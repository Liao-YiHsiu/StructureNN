#ifndef _MYNNET_LOSS_H_
#define _MYNNET_LOSS_H_

#include "nnet/nnet-nnet.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "nnet/nnet-activation.h"
#include "util.h"
#include "strt.h"
#include <boost/algorithm/string.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <iterator>
#include <sstream>
#include <algorithm>

using namespace std;
using namespace kaldi;
using namespace kaldi::nnet1;
using namespace boost::algorithm;

class LabelLossBase{
   public:
      typedef enum{
         lUnknown = 0x0,

         lList    = 0x0100,

         lFrame   = 0x0200,

         lMulti   = 0x0300
      } MyLossType;

      struct my_key_value{
         const MyLossType key;
         const char* value;
      };

      static const struct my_key_value myMarkerMap[];
      static const char*  myTypeToMarker(MyLossType t);
      static MyLossType   myMarkerToType(const string &s);

      static LabelLossBase* Read(const string &file);
      static LabelLossBase* GetInstance(const string &conf_line);


      LabelLossBase() {}
      virtual ~LabelLossBase() {}

      virtual MyLossType GetType() = 0;

      virtual void Eval(const vector< vector<uchar> > &labels, const CuMatrixBase<BaseFloat> &nnet_out, 
            CuMatrix<BaseFloat> *nnet_out_diff) = 0;

      virtual string Report() { return ""; }

      virtual void SetParam(istream &is) = 0;

   protected:
      static LabelLossBase* NewMyLossOfType(MyLossType type);
};

class LabelListLoss : public LabelLossBase{
   public:
      LabelListLoss() : strt_(NULL) {}
      virtual ~LabelListLoss();
      
      virtual MyLossType GetType() { return lList; }

      virtual void SetParam(istream &is);

      virtual void Eval(const vector< vector<uchar> > &labels, const CuMatrixBase<BaseFloat> &nnet_out, 
            CuMatrix<BaseFloat> *nnet_out_diff);

      virtual string Report() { return strt_->Report(); }

   private:
      StrtListBase *strt_;
};

class LabelFrameLoss : public LabelLossBase{
   public:
      LabelFrameLoss():softmax(2,2){}
      virtual ~LabelFrameLoss() {}

      virtual MyLossType GetType() { return lFrame; }

      virtual void SetParam(istream &is) {}

      virtual void Eval(const vector< vector<uchar> > &labels, const CuMatrixBase<BaseFloat> &nnet_out, 
            CuMatrix<BaseFloat> *nnet_out_diff);

      virtual string Report() { return xent.Report(); }

   private:
      Xent xent;
      Softmax softmax;
};

class LabelMultiLoss : public LabelLossBase{
   public:
      LabelMultiLoss(vector<LabelLossBase*> loss_arr, vector<BaseFloat> loss_weight) :
         loss_arr_(loss_arr), loss_weight_(loss_weight) {}
      virtual ~LabelMultiLoss();

      virtual MyLossType GetType() { return lMulti; }

      virtual void SetParam(istream &is) {}

      virtual void Eval(const vector< vector<uchar> > &labels, const CuMatrixBase<BaseFloat> &nnet_out, 
            CuMatrix<BaseFloat> *nnet_out_diff);

      virtual string Report();
   private:
      vector<LabelLossBase*> loss_arr_;
      vector<BaseFloat>      loss_weight_;
};

#endif
