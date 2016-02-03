#ifndef _MY_CU_UTILS_H_
#define _MY_CU_UTILS_H_

#include "nnet/nnet-nnet.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "my-cumatrix/cu-matrix.h"
#include <string>
#include <vector>

using namespace std;
using namespace kaldi;
using namespace kaldi::nnet1;

void VecToMat(const vector< CuMatrix<BaseFloat> > &arr, CuMatrix<BaseFloat> &mat, int N = -1);
void RepMat(const CuMatrix<BaseFloat> &src, CuMatrix<BaseFloat> &dest, int N = -1);

void MatToVec(const CuMatrix<BaseFloat> &mat, const vector< CuMatrix<BaseFloat> > &ref,
      vector< CuMatrix<BaseFloat> > &arr, int N = -1);
void MatToVec(const CuMatrix<BaseFloat> &mat, const vector<int> &ref,
      vector< CuMatrix<BaseFloat> > &arr);

vector<int> getRowsN(const vector< CuMatrix<BaseFloat> > &arr);

void Sum(const vector< CuMatrix<BaseFloat> > &arr, CuMatrix<BaseFloat>* out, int N = -1);

bool Same(const CuMatrixBase<BaseFloat> &a, const CuMatrixBase<BaseFloat> &b, double err = 1e-8);

void getCuData(vector< MyCuMatrix<BaseFloat> > &in_arr, vector<BaseFloat*> &ptr_arr, vector<int32> &stride_arr);

#endif
