#ifndef _NNET_BATCH_H_
#define _NNET_BATCH_H_

#include "nnet/nnet-nnet.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "util.h"
#include <iostream>
#include <string>
#include <vector>

using namespace std;
using namespace kaldi;
using namespace kaldi::nnet1;

class BNnet : public Nnet{
   public:
      BNnet() {}
      BNnet(const Nnet& other) : Nnet(other) {}
      BNnet(const BNnet& other) : Nnet(other) {}
      BNnet &operator = (const BNnet &other) {Nnet::operator=(other); return *this;}

      ~BNnet() {}

   public:
      void Propagate(const vector< CuMatrix<BaseFloat> > &in_arr,
            vector< CuMatrix<BaseFloat> > &out_arr);

      void Backpropagate(const vector< CuMatrix<BaseFloat> > &out_diff, 
            vector< CuMatrix<BaseFloat> > *in_diff = NULL);
      
      void Backpropagate(const CuMatrix<BaseFloat> &out_diff, 
            vector< CuMatrix<BaseFloat> > *in_diff = NULL, int N = -1);

      void Feedforward(const vector< CuMatrix<BaseFloat> > &in_arr,
            vector< CuMatrix<BaseFloat> > &out_arr); 

      // original
      void Propagate(const CuMatrixBase<BaseFloat> &in, CuMatrix<BaseFloat> *out); 
      void Backpropagate(const CuMatrixBase<BaseFloat> &out_diff, CuMatrix<BaseFloat> *in_diff);
      void Feedforward(const CuMatrixBase<BaseFloat> &in, CuMatrix<BaseFloat> *out); 

   private:
      void VecToMat(const vector< CuMatrix<BaseFloat> > &arr, CuMatrix<BaseFloat> &mat);
      void MatToVec(const CuMatrix<BaseFloat> &mat, const vector< CuMatrix<BaseFloat> > &ref,
            vector< CuMatrix<BaseFloat> > &arr);
      void MatToVec(const CuMatrix<BaseFloat> &mat, const vector<int> &ref,
            vector< CuMatrix<BaseFloat> > &arr);
      void RepMat(const CuMatrix<BaseFloat> &src, CuMatrix<BaseFloat> &dest, int N);
      vector<int> getRowsN(const vector< CuMatrix<BaseFloat> > &arr);
};

#endif // _NNET_BATCH_H_
