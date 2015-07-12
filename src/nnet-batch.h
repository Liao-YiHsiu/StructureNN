#ifndef _NNET_BATCH_H_
#define _NNET_BATCH_H_

#include "nnet/nnet-nnet.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "nnet-cache.h"
#include <iostream>
#include <string>
#include <vector>

using namespace std;
using namespace kaldi;
using namespace kaldi::nnet1;

class BNnet : public Nnet{
   public:
      BNnet() {}
      BNnet(const BNnet& other) : Nnet(other) {}
      BNnet &operator = (const BNnet &other) {Nnet::operator=(other); return *this;}

      ~BNnet() {}

   public:
      void Propagate(const vector< CuMatrix<BaseFloat> > &in_arr,
            vector< CuMatrix<BaseFloat> > &out_arr);

      void Backpropagate(const vector< CuMatrix<BaseFloat> > &out_diff);

      void Feedforward(const vector< CuMatrix<BaseFloat> > &in_arr,
            vector< CuMatrix<BaseFloat> > &out_arr); 

      void Feedforward(const CuMatrix<BaseFloat> &in, CuMatrix<BaseFloat> *out); 

   private:
      void VecToMat(const vector< CuMatrix<BaseFloat> > &arr, CuMatrix<BaseFloat> &mat);
      void MatToVec(const CuMatrix<BaseFloat> &mat, const vector< CuMatrix<BaseFloat> > &ref,
            vector< CuMatrix<BaseFloat> > &arr);
};

#endif // _NNET_BATCH_H_
