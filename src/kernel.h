#ifndef MY_KERNEL
#define MY_KERNEL

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

#endif
