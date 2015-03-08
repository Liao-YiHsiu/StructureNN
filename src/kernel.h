#ifndef MY_KERNEL
#define MY_KERNEL

#define BLOCKSIZE 64 


void cuda_make_obs(dim3 grid, dim3 block, const float* feats, int rows, int cols, int stride, const int* lab, float *data, int d_stride, int S);

void cuda_make_tran(dim3 grid, dim3 block, int rows, int cols, const int* lab, float *data, int d_stride, int S);

void cuda_find_max(dim3 grid, dim3 block, int sharemem, const float* arr, int len, float *val, int* index);

void cuda_make_obs(dim3 grid, dim3 block, const float* feats, int rows, int cols, int stride, const int* lab, int l, float *data, int d_stride, int S);

void cuda_make_tran(dim3 grid, dim3 block, int rows, int cols, const int* lab, int l, float *data, int d_stride, int S);

#endif
