#ifndef MY_KERNEL
#define MY_KERNEL

#include "my-utils/type.h"

#define BLOCKSIZE 64 

typedef struct{
   int           L;
   int           feat_stride;

   unsigned char *lab;
   float         *feat;
   float         *psi_feat;
} PsiPack;

void cuda_make_obs(dim3 grid, dim3 block, const float* feats, int rows, int cols, int stride, const int* lab, float *data, int d_stride, int S);

void cuda_make_tran(dim3 grid, dim3 block, int rows, int cols, const int* lab, float *data, int d_stride, int S);

void cuda_find_max(dim3 grid, dim3 block, int sharemem, const float* arr, int len, float *val, int* index);

void cuda_make_obs(dim3 grid, dim3 block, const float* feats, int rows, int cols, int stride, const int* lab, int l, float *data, int d_stride, int S);

void cuda_make_tran(dim3 grid, dim3 block, int rows, int cols, const int* lab, int l, float *data, int d_stride, int S);

void cuda_prop_psi(dim3 grid, dim3 block, size_t shared_mem, int N, int F, int S, PsiPack *packs_ptr);

void cuda_back_psi(dim3 grid, dim3 block, size_t shared_mem, int N, int F, int S, PsiPack *packs_ptr);

void cuda_distribute(dim3 grid, dim3 block, const float* mat, int rows, int cols, int stride,
      const int* seq_arr, const int* id_arr, float** mat_arr);

void cuda_combine(dim3 grid, dim3 block, float* mat, int rows, int cols, int stride,
      const int* seq_arr, const int* id_arr, const float** mat_arr);


void cuda_dist_prop(dim3 grid, dim3 block, const float* mat, int rows, int cols, int stride, 
      const int* seq_arr, int seq_stride, const int* id_arr, float** mat_arr, int* mat_arr_stride);

void cuda_comb_prop(dim3 grid, dim3 block, float* mat, int rows, int cols, int stride, 
      const int* seq_arr, int seq_stride, const int* id_arr, float** mat_arr, int* mat_arr_stride);

void cuda_dist_back(dim3 grid, dim3 block, const float* mat, int rows, int cols, int stride, 
      const int* seq_arr, int seq_stride, const int* id_arr, float** mat_arr, int* mat_arr_stride);

void cuda_comb_back(dim3 grid, dim3 block, float* mat, int rows, int cols, int stride, 
      const int* seq_arr, int seq_stride, const int* id_arr, float** mat_arr, int* mat_arr_stride);

void cuda_embed_prop(dim3 grid, dim3 block, const float* mat, int rows, int cols, int stride,
      const uchar* seq_arr, int seq_stride, float* out_mat, int out_rows, int out_stride);

void cuda_embed_back(dim3 grid, dim3 block, const float* mat, int rows, int stride, int seq_stride,
      float *out_mat, int out_rows, int out_cols, int out_stride);

void cuda_blendsum_prop(dim3 grid, dim3 block, const float* mat, int rows, int cols, int stride, 
      const int* seq_arr, int seq_size, float* out_mat, int out_rows, int out_stride);

void cuda_blendsum_back(dim3 grid, dim3 block, const float* mat, int rows, int cols, int stride,
      const int* seq_arr, int seq_size, float* out_mat, int out_rows, int out_stride);

void cuda_mem_copy(dim3 grid, dim3 block, float* dst, int dst_pitch, const float* src, int src_pitch,
     int width, int height);

typedef struct{
   int           L;            // label #
   int           T;            // utterance length
   int           P;            // phone max state
   int           D;            // D

   int           phone_feat_stride;
   int           frame_feat_stride;

   unsigned char *lab;         // lab[l*T + t] == lab[l][t]
   float         **phone_feat; // phone_feat[p] -> a matrix [t][d] phone_feat[p][t*phone_feat_stride + d]
   float         **frame_feat; // frame_feat[t] -> a matrix [l][d] frame_feat[t][l*frame_feat_stride + d]
} RPsiPack;

void cuda_prop_rpsi(dim3 grid, dim3 block, RPsiPack *pack);
void cuda_back_rpsi(dim3 grid, dim3 block, size_t shared_mem, RPsiPack *pack);

#endif
