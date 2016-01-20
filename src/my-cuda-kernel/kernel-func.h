#ifndef _MY_KERNEL_FUNC_H_
#define _MY_KERNEL_FUNC_H_

#include "cudamatrix/cu-matrix.h"
#include "my-cuda-kernel/kernel.h"
#include <vector>

using namespace std;
using namespace kaldi;

void propPsi(int N, int F, int S, int maxL, PsiPack* packs_ptr);
void backPsi(int N, int F, int S, int maxL, PsiPack* packs_ptr);

void propRPsi(RPsiPack *pack);
void backRPsi(RPsiPack *pack);

void dist_prop(const CuMatrixBase<BaseFloat> &mat, const int* seq_arr, int seq_stride,
      const int* id_arr, float** mat_arr, int* mat_arr_stride);

void comb_prop(float** mat_arr, int* mat_arr_stride, const int* seq_arr, int seq_stride,
      const int* id_arr, CuMatrixBase<BaseFloat> &mat);

void dist_back(const CuMatrixBase<BaseFloat> &mat, const int* seq_arr, int seq_stride,
      const int* id_arr, float** mat_arr, int* mat_arr_stride);

void comb_back(float** mat_arr, int* mat_arr_stride, const int* seq_arr, int seq_stride,
      const int* id_arr, CuMatrixBase<BaseFloat> &mat);

void embed_prop(const CuMatrixBase<BaseFloat> &in, const uchar* seq_arr, int seq_stride, 
      CuMatrixBase<BaseFloat> &out);

void embed_back(const CuMatrixBase<BaseFloat> &out_diff, int seq_stride, 
      CuMatrixBase<BaseFloat> &in_diff);

void blendsum_prop(const CuMatrixBase<BaseFloat> &in, const int* seq_arr, int seq_size,
      CuMatrixBase<BaseFloat> &out);

void blendsum_back(const CuMatrixBase<BaseFloat> &out_diff, const int *seq_arr, int seq_size,
      CuMatrixBase<BaseFloat> &in_diff);

void cuMemCopy(float* dst, int dst_pitch,const float* src, int src_pitch, int width, int height);

void fillin(CuMatrixBase<BaseFloat> &dest, vector< CuMatrix<BaseFloat> > &src, int stream_num);

#endif
