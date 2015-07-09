#ifndef _NNET_CACHE_H_
#define _NNET_CACHE_H_

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

// an Nnet with buffer
class CNnet : public Nnet{
   public:
      CNnet(){}
      CNnet(const CNnet &other) : Nnet(other) {}
      CNnet &operator = (const CNnet& other) {Nnet::operator=(other); return *this;}

      ~CNnet();

   public:
      void Propagate(const CuMatrixBase<BaseFloat> &in, CuMatrix<BaseFloat> *out, int index);
      void Backpropagate(const CuMatrixBase<BaseFloat> &out_diff, CuMatrix<BaseFloat> *in_diff, int index);
      void Update();

      void GetComponents(vector<Component*> &components);

      void Check() const;
      void Destroy();

   private:
      // resize all propagate_buf_arr[i] and back_propagate_buf_arr[i]
      void resizeAll();
      
      vector< vector<CuMatrix<BaseFloat> > > propagate_buf_arr_;
      vector< vector<CuMatrix<BaseFloat> > > backpropagate_buf_arr_;
};

#endif // _NNET_CACHE_H_
