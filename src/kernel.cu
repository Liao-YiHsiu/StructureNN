#include "kernel.h"
#include <assert.h>
#define MAX_L 1024
#define MAX_S 64

__device__ static void _load(int* tgt, const int* src, int L, int l, int s);

__global__ static void _cuda_make_obs(const float* feats, int rows, int cols, int stride, const int* lab, float *data, int d_stride, int S);

__global__ static void _cuda_make_tran(int rows, int cols, const int* lab, float *data, int d_stride, int S);

__global__ static void _cuda_make_obs(const float* feats, int rows, int cols, int stride, const int* lab, int l, float *data, int d_stride, int S);

__global__ static void _cuda_make_tran(int rows, int cols, const int* lab, int l, float *data, int d_stride, int S);

__global__ static void _cuda_find_max(const float* arr, int len, float *val, int* index);

__global__ static void _cuda_prop_psi(int N, int F, int S, PsiPack *packs_ptr);

__global__ static void _cuda_back_psi(int N, int F, int S, PsiPack *packs_ptr);

__global__ static void _cuda_prop_rpsi(RPsiPack pack);

__global__ static void _cuda_back_rpsi(RPsiPack pack);

__global__ static void _cuda_distribute(const float* mat, int rows, int cols, int stride, 
      const int* seq_arr, const int* id_arr, float** mat_arr);

__global__ static void _cuda_combine(float* mat, int rows, int cols, int stride,
      const int* seq_arr, const int* id_arr, const float** mat_arr);


__global__ static void _cuda_dist_prop(const float* mat, int rows, int cols, int stride,
      const int* seq_arr, int seq_stride, const int* id_arr, float** mat_arr, int* mat_arr_stride);

__global__ static void _cuda_comb_prop(float* mat, int rows, int cols, int stride,
      const int* seq_arr, int seq_stride, const int* id_arr, float** mat_arr, int* mat_arr_stride);

__global__ static void _cuda_dist_back(const float* mat, int rows, int cols, int stride,
      const int* seq_arr, int seq_stride, const int* id_arr, float** mat_arr, int* mat_arr_stride);

__global__ static void _cuda_comb_back(float* mat, int rows, int cols, int stride,
      const int* seq_arr, int seq_stride, const int* id_arr, float** mat_arr, int* mat_arr_stride);

void cuda_make_obs(dim3 grid, dim3 block, const float* feats, int rows, int cols, int stride, const int* lab, float *data, int d_stride, int S){
   assert( rows < MAX_L);
   assert( S < MAX_S);
   _cuda_make_obs<<<grid, block>>>(feats, rows, cols, stride, lab, data, d_stride, S);
}

void cuda_make_tran(dim3 grid, dim3 block, int rows, int cols, const int* lab, float *data, int d_stride, int S){
   assert( rows < MAX_L);
   assert( S < MAX_S);
   _cuda_make_tran<<<grid, block>>>(rows, cols, lab, data, d_stride, S);
}

void cuda_find_max(dim3 grid, dim3 block, int sharemem, const float* arr, int len, float *val, int* index){
   _cuda_find_max<<<grid, block, sharemem>>>(arr, len, val, index);
}


void cuda_make_obs(dim3 grid, dim3 block, const float* feats, int rows, int cols, int stride, const int* lab, int l, float *data, int d_stride, int S){
   assert( rows < MAX_L);
   assert( S < MAX_S);
   _cuda_make_obs<<<grid, block>>>(feats, rows, cols, stride, lab, l, data, d_stride, S);
}

void cuda_make_tran(dim3 grid, dim3 block, int rows, int cols, const int* lab, int l, float *data, int d_stride, int S){
   assert( rows < MAX_L);
   assert( S < MAX_S);
   _cuda_make_tran<<<grid, block>>>(rows, cols, lab, l, data, d_stride, S);
}

void cuda_prop_psi(dim3 grid, dim3 block, size_t shared_mem, int N, int F, int S, PsiPack *packs_ptr){
   _cuda_prop_psi<<<grid, block, shared_mem>>>(N, F, S, packs_ptr);
}

void cuda_back_psi(dim3 grid, dim3 block, size_t shared_mem, int N, int F, int S, PsiPack *packs_ptr){
   _cuda_back_psi<<<grid, block, shared_mem*sizeof(float)>>>(N, F, S, packs_ptr);
}

void cuda_prop_rpsi(dim3 grid, dim3 block, RPsiPack *pack){
   _cuda_prop_rpsi<<<grid, block>>>(*pack);
}

void cuda_back_rpsi(dim3 grid, dim3 block, size_t shared_mem, RPsiPack *pack){
   _cuda_back_rpsi<<<grid, block, shared_mem*sizeof(unsigned char)>>>(*pack);
}

void cuda_distribute(dim3 grid, dim3 block, const float* mat, int rows, int cols, int stride,
      const int* seq_arr, const int* id_arr, float** mat_arr){
   _cuda_distribute<<<grid, block>>>(mat, rows, cols, stride, seq_arr, id_arr, mat_arr);
}

void cuda_combine(dim3 grid, dim3 block, float* mat, int rows, int cols, int stride,
      const int* seq_arr, const int* id_arr, const float** mat_arr){
   _cuda_combine<<<grid, block>>>(mat, rows, cols, stride, seq_arr, id_arr, mat_arr);
}


void cuda_dist_prop(dim3 grid, dim3 block, const float* mat, int rows, int cols, int stride, 
      const int* seq_arr, int seq_stride, const int* id_arr, float** mat_arr, int* mat_arr_stride){
   _cuda_dist_prop<<<grid, block>>>(mat, rows, cols, stride, seq_arr, seq_stride, id_arr, mat_arr, mat_arr_stride);
}

void cuda_comb_prop(dim3 grid, dim3 block, float* mat, int rows, int cols, int stride, 
      const int* seq_arr, int seq_stride, const int* id_arr, float** mat_arr, int* mat_arr_stride){
   _cuda_comb_prop<<<grid, block>>>(mat, rows, cols, stride, seq_arr, seq_stride, id_arr, mat_arr, mat_arr_stride);
}

void cuda_dist_back(dim3 grid, dim3 block, const float* mat, int rows, int cols, int stride, 
      const int* seq_arr, int seq_stride, const int* id_arr, float** mat_arr, int* mat_arr_stride){
   _cuda_dist_back<<<grid, block>>>(mat, rows, cols, stride, seq_arr, seq_stride, id_arr, mat_arr, mat_arr_stride);
}

void cuda_comb_back(dim3 grid, dim3 block, float* mat, int rows, int cols, int stride, 
      const int* seq_arr, int seq_stride, const int* id_arr, float** mat_arr, int* mat_arr_stride){
   _cuda_comb_back<<<grid, block>>>(mat, rows, cols, stride, seq_arr, seq_stride, id_arr, mat_arr, mat_arr_stride);
}

__device__
static void _load(int* tgt, const int* src, int L, int l, int s){
   for(int i = 0; i < L; i++)
      tgt[i] = src[i];
   tgt[l] = s;
}

__global__
static void _cuda_make_obs(const float* feats, int rows, int cols, int stride,
      const int* lab, float *data, int d_stride, int S){

   int L = rows;
   int F = cols;

   // change lab[l] = s, and feats[c]
   int idx = blockIdx.x * blockDim.x + threadIdx.x; 

   int s = idx % S; idx /= S;
   int l = idx % L; idx /= L;
   int f = idx;

   if(f >= F) return;


   // TODO
   // load label into share memory
   // load feats

   int mylab[MAX_L];
   _load(mylab, lab, L, l, s);

   float tmparr[MAX_S];
   for(int i = 0; i < S; ++i)
      tmparr[i] = 0;

   for(int i = 0; i < L; ++i){
      tmparr[mylab[i]] += feats[i * stride + f];
   }

   float *tgt = data + (l*S + s)*d_stride;

   for(int i = 0; i < S; ++i){
      tgt[i*F + f] = tmparr[i]/L;
   }


}

__global__
static void _cuda_make_tran(int rows, int cols, const int* lab, float *data, int d_stride, int S){

   int L = rows;
   int F = cols;

   // change lab[l] = s, and feats[c]
   int idx = blockIdx.x * blockDim.x + threadIdx.x; 

   int l = idx % L; idx /= L;
   int s = idx;
   
   if( s >= S ) return;


   // TODO
   // load label into share memory
   // load feats

   int mylab[MAX_L];
   _load(mylab, lab, L, l, s);

   float *tgt = data + (l*S + s)*d_stride + S*F;

   float one = 1/(float)L;
   for(int i = 1; i < L; ++i)
      tgt[mylab[i - 1]*S + mylab[i]] += one;

}



__global__
static void _cuda_make_obs(const float* feats, int rows, int cols, int stride,
      const int* lab, int l, float *data, int d_stride, int S){

   int L = rows;
   int F = cols;

   // change lab[l] = s, and feats[c]
   int idx = blockIdx.x * blockDim.x + threadIdx.x; 

   int s = idx % S; idx /= S;
   int f = idx;

   if(f >= F) return;


   // TODO
   // load label into share memory
   // load feats

   int mylab[MAX_L];
   _load(mylab, lab, L, l, s);

   float tmparr[MAX_S];
   for(int i = 0; i < S; ++i)
      tmparr[i] = 0;

   for(int i = 0; i < L; ++i){
      tmparr[mylab[i]] += feats[i * stride + f];
   }

   float *tgt = data + s*d_stride;

   for(int i = 0; i < S; ++i){
      tgt[i*F + f] = tmparr[i]/L;
   }


}

__global__
static void _cuda_make_tran(int rows, int cols, const int* lab, int l, float *data, int d_stride, int S){

   int L = rows;
   int F = cols;

   // change lab[l] = s, and feats[c]
   int idx = blockIdx.x * blockDim.x + threadIdx.x; 
   int s = idx;
   

   if( s >= S ) return;


   // TODO
   // load label into share memory
   // load feats

   int mylab[MAX_L];
   _load(mylab, lab, L, l, s);

   float *tgt = data + s*d_stride + S*F;

   float one = 1/(float)L;
   for(int i = 1; i < L; ++i)
      tgt[mylab[i - 1]*S + mylab[i]] += one;

}

__global__
static void _cuda_find_max(const float* arr, int len, float *val, int* index){
   extern __shared__ float share[];

   float* sdata = (float*)share;
   int*   idx   = (int* )&sdata[blockDim.x];


   int tid = threadIdx.x;
   int i = blockIdx.x * blockDim.x + threadIdx.x;


   if(i < len){
      sdata[tid] = arr[i];
      idx[tid]   = i;
   }else{
      sdata[tid] = 0;
      idx[tid]   = -1;
   }
   __syncthreads();

   //if(i >= len) return;

   for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
      if (tid < s) {
         if(sdata[tid] < sdata[tid + s]){
            sdata[tid] = sdata[tid + s];
            idx[tid] = idx[tid + s];
         }
      }
      __syncthreads();
   }

   if(tid == 0){
      val[blockIdx.x] = sdata[0];
      index[blockIdx.x] = idx[0];
   }

}

__global__ static void _cuda_prop_psi(int N, int F, int S, PsiPack *packs_ptr){

   extern __shared__ unsigned char label[];

   int n    = blockIdx.x;  // N            ( 0 <= n    < N)
   int cols = threadIdx.x; // columns of F ( 0 <= cols < F)

   PsiPack &pack   = packs_ptr[n];

   int L           = pack.L;
   int feat_stride = pack.feat_stride;
   float *feat     = pack.feat;
   float *psi_feat = pack.psi_feat;

   // move lables into shared memory.
   for(int i = cols; i < L; i += F)
      label[i] = pack.lab[i];
   __syncthreads();

   for(int i = 0; i < L; ++i){
      float value = feat[i * feat_stride + cols];
      psi_feat[cols] += value;
      psi_feat[F * label[i] + cols] += value;
   }
}

// TODO change threads from feat[idx] to fully workable units
__global__ static void _cuda_back_psi(int N, int F, int S, PsiPack *packs_ptr){

   extern __shared__ float psi_feat[];

   int n   = blockIdx.x;  // N            ( 0 <= n    < N)
   int idx = threadIdx.x; // feat[idx]    ( 0 <= idx  < maxL)

   PsiPack &pack   = packs_ptr[n];

   int L           = pack.L;
   int feat_stride = pack.feat_stride;
   float *feat     = pack.feat;
   int psi_dim     = F + F * S;

   // move lables into shared memory.
   for(int i = idx; i < psi_dim; i += blockDim.x)
      psi_feat[i] = pack.psi_feat[i];
   __syncthreads();

   if( idx >= L ) return;

   unsigned char lab = pack.lab[idx];
   
   if(lab == 0) return;

   for(int i = 0; i < F; ++i){
      feat[idx*feat_stride + i] = psi_feat[i] + psi_feat[F*lab + i];
   }
}

__global__ static void _cuda_prop_rpsi(RPsiPack pack){
   int t = blockIdx.x;
   int l = threadIdx.x;

   int T = pack.T;
   int D = pack.D;
   
   int phone_feat_stride = pack.phone_feat_stride;
   int frame_feat_stride = pack.frame_feat_stride;

   unsigned char lab = pack.lab[l*T + t];

   float * frame = pack.frame_feat[t] + l * frame_feat_stride;
   float * phone = pack.phone_feat[lab - 1] + t * phone_feat_stride;

   if(lab == 0)return;
   for(int i = 0; i < D; ++i)
      frame[i] = phone[i];
}

__global__ static void _cuda_back_rpsi(RPsiPack pack){

   extern __shared__ unsigned char lab[];

   int t = blockIdx.x;
   int p = threadIdx.x;

   int L = pack.L;
   int T = pack.T;
   int D = pack.D;
   
   int phone_feat_stride = pack.phone_feat_stride;
   int frame_feat_stride = pack.frame_feat_stride;
   
   // moving data to label. for the same time t.
   for(int l = 0; l < L; ++l)
      lab[l] = pack.lab[l*T + t];

   float * phone = pack.phone_feat[ p ] + t * phone_feat_stride;
   float * frame = pack.frame_feat[ t ];

   for(int l = 0; l < L; ++l)
      if(p == lab[l] - 1){
         float * frm = frame + l * frame_feat_stride;
         for(int d = 0; d < D; ++d){
            // TODO use temporal mem.
            phone[d] += frm[d];
         }
      }
}

__global__ static void _cuda_distribute(const float* mat, int rows, int cols, int stride,
      const int* seq_arr, const int* id_arr, float** mat_arr){

   int idx = blockIdx.x * blockDim.x + threadIdx.x;

   if(idx >= rows) return;

   float       *odata = mat_arr[seq_arr[idx]] + id_arr[idx] * stride;
   const float *idata = mat + stride * idx;

   for(int i = 0; i < cols; ++i)
      odata[i] = idata[i];
}

__global__ static void _cuda_combine(float* mat, int rows, int cols, int stride,
      const int* seq_arr, const int* id_arr, const float** mat_arr){

   int idx = blockIdx.x * blockDim.x + threadIdx.x;

   if(idx >= rows) return;

   const float *idata = mat_arr[seq_arr[idx]] + id_arr[idx] * stride;
   float       *odata = mat + stride * idx;

   for(int i = 0; i < cols; ++i)
      odata[i] = idata[i];
}

__global__ static void _cuda_dist_prop(const float* mat, int rows, int cols, int stride,
      const int* seq_arr, int seq_stride, const int* id_arr, float** mat_arr, int* mat_arr_stride){

   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   if(idx >= rows) return;

   const int *seqs = seq_arr + seq_stride * idx;
   const int *ids  = id_arr  + seq_stride * idx;

   char mask[64] = {0};
   
   const float *idata = mat + stride * idx;

   for(int i = 0; i < seq_stride; ++i){
      if(mask[seqs[i]]) continue;
      float *odata = mat_arr[seqs[i]] + ids[i] * mat_arr_stride[seqs[i]];

      for(int j = 0; j < cols; ++j)
         odata[j] = idata[j];
      mask[seqs[i]] = 1;
   }
}

__global__ static void _cuda_comb_prop(float* mat, int rows, int cols, int stride,
      const int* seq_arr, int seq_stride, const int* id_arr, float** mat_arr, int* mat_arr_stride){
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   if(idx >= rows) return;

   const int *seqs = seq_arr + seq_stride * idx;
   const int *ids  = id_arr  + seq_stride * idx;

   for(int i = 0; i < seq_stride; ++i){
      float *idata = mat_arr[seqs[i]] + ids[i] * mat_arr_stride[seqs[i]];
      float *odata = mat + stride * (idx * seq_stride + i);

      for(int j = 0; j < cols; ++j)
         odata[j] = idata[j];
   }
}

__global__ static void _cuda_dist_back(const float* mat, int rows, int cols, int stride,
      const int* seq_arr, int seq_stride, const int* id_arr, float** mat_arr, int* mat_arr_stride){
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   if(idx >= rows) return;

   const int *seqs = seq_arr + seq_stride * idx;
   const int *ids  = id_arr  + seq_stride * idx;

   for(int i = 0; i < seq_stride; ++i){
      float       *odata = mat_arr[seqs[i]] + ids[i] * mat_arr_stride[seqs[i]];
      const float *idata = mat + stride * (idx * seq_stride + i);

      for(int j = 0; j < cols; ++j)
         odata[j] += idata[j];
   }
}

__global__ static void _cuda_comb_back(float* mat, int rows, int cols, int stride,
      const int* seq_arr, int seq_stride, const int* id_arr, float** mat_arr, int* mat_arr_stride){

   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   if(idx >= rows) return;

   const int *seqs = seq_arr + seq_stride * idx;
   const int *ids  = id_arr  + seq_stride * idx;

   char mask[64] = {0};

   float *odata = mat + stride * idx;

   for(int i = 0; i < seq_stride; ++i){
      if(mask[seqs[i]]) continue;
      float *idata = mat_arr[seqs[i]] + ids[i] * mat_arr_stride[seqs[i]];

      for(int j = 0; j < cols; ++j)
         odata[j] += idata[j];
      mask[seqs[i]] = 1;
   }
}
