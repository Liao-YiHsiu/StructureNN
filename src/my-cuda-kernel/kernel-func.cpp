#include "my-cuda-kernel/kernel-func.h"

// N = # of packs_ptr, F = dimension of feats, S = max state.
void propPsi(int N, int F, int S, int maxL, PsiPack* packs_ptr){
   Timer tim;

   // TODO compute the dummy part only once.
   cuda_prop_psi(N, F, maxL, N, F, S, packs_ptr);

   CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
}

// N = # of packs_ptr, F = dimension of feats, S = max state.
void backPsi(int N, int F, int S, int maxL, PsiPack* packs_ptr){
   Timer tim;

   // TODO compute the dummy part only once.
   cuda_back_psi(N, maxL, F*S + F, N, F, S, packs_ptr);

   CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
}

void propRPsi(RPsiPack* pack){
   Timer tim;

   cuda_prop_rpsi(pack->T, pack->L, pack);

   CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
}

void backRPsi(RPsiPack *pack){
   Timer tim;

   cuda_back_rpsi(pack->T, pack->P, pack->L, pack);

   CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
}

void dist_prop(const CuMatrixBase<BaseFloat> &mat, const int* seq_arr, int seq_stride,
      const int* id_arr, float** mat_arr, int* mat_arr_stride){
   Timer tim;

   int rows = mat.NumRows();

   cuda_dist_prop((rows-1)/BLOCKSIZE+1, BLOCKSIZE, mat.Data(),
         rows, mat.NumCols(), mat.Stride(), seq_arr, seq_stride, id_arr, mat_arr, mat_arr_stride);

   CU_SAFE_CALL(cudaGetLastError()); 
   CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
}

void comb_prop(float** mat_arr, int* mat_arr_stride, const int* seq_arr, int seq_stride,
      const int* id_arr, CuMatrixBase<BaseFloat> &mat){
   Timer tim;

   int rows = mat.NumRows()/seq_stride;

   cuda_comb_prop((rows-1)/BLOCKSIZE+1, BLOCKSIZE, mat.Data(),
         rows, mat.NumCols(), mat.Stride(), seq_arr, seq_stride, id_arr, mat_arr, mat_arr_stride);

   CU_SAFE_CALL(cudaGetLastError()); 
   CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
}

void dist_back(const CuMatrixBase<BaseFloat> &mat, const int* seq_arr, int seq_stride,
      const int* id_arr, float** mat_arr, int* mat_arr_stride){
   Timer tim;

   int rows = mat.NumRows()/seq_stride;

   cuda_dist_back((rows-1)/BLOCKSIZE+1, BLOCKSIZE, mat.Data(),
         rows, mat.NumCols(), mat.Stride(), seq_arr, seq_stride, id_arr, mat_arr, mat_arr_stride);

   CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
}

void comb_back(float** mat_arr, int* mat_arr_stride, const int* seq_arr, int seq_stride,
      const int* id_arr, CuMatrixBase<BaseFloat> &mat){
   Timer tim;

   int rows = mat.NumRows();

   cuda_comb_back((rows-1)/BLOCKSIZE+1, BLOCKSIZE, mat.Data(),
         rows, mat.NumCols(), mat.Stride(), seq_arr, seq_stride, id_arr, mat_arr, mat_arr_stride);

   CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
}

void embed_prop(const CuMatrixBase<BaseFloat> &in, const uchar* seq_arr, int seq_stride, 
      CuMatrixBase<BaseFloat> &out){
   Timer tim;

   int rows = out.NumRows();

   cuda_embed_prop((rows-1)/BLOCKSIZE+1, BLOCKSIZE,
         in.Data(), in.NumRows(), in.NumCols(), in.Stride(),
         seq_arr, seq_stride, 
         out.Data(), rows, out.Stride());

   CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
}

void embed_back(const CuMatrixBase<BaseFloat> &out_diff, int seq_stride, 
      CuMatrixBase<BaseFloat> &in_diff){
   Timer tim;

   int threads = in_diff.NumRows() * in_diff.NumCols();

   cuda_embed_back((threads-1)/BLOCKSIZE+1, BLOCKSIZE,
         out_diff.Data(), out_diff.NumRows(), out_diff.Stride(), seq_stride,
         in_diff.Data(), in_diff.NumRows(), in_diff.NumCols(), in_diff.Stride());

   CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
}

void blendsum_prop(const CuMatrixBase<BaseFloat> &in, const int* seq_arr, int seq_size,
      CuMatrixBase<BaseFloat> &out){
   Timer tim;
   
   int threads = out.NumRows() * out.NumCols();

   cuda_blendsum_prop((threads-1)/BLOCKSIZE+1, BLOCKSIZE,
         in.Data(), in.NumRows(), in.NumCols(), in.Stride(),
         seq_arr, seq_size, out.Data(), out.NumRows(), out.Stride());

   CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
}

void blendsum_back(const CuMatrixBase<BaseFloat> &out_diff, const int *seq_arr, int seq_size,
      CuMatrixBase<BaseFloat> &in_diff){
   Timer tim;
   
   int threads = out_diff.NumRows() * out_diff.NumCols();

   cuda_blendsum_back((threads-1)/BLOCKSIZE+1, BLOCKSIZE,
         out_diff.Data(), out_diff.NumRows(), out_diff.NumCols(), out_diff.Stride(),
         seq_arr, seq_size, in_diff.Data(), in_diff.NumRows(), in_diff.Stride());

   CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
}

void cuMemCopy(float* dst, int dst_pitch, const float* src, int src_pitch, int width, int height){
   Timer tim;

   int threads = width * height;

   cuda_mem_copy((threads-1)/BLOCKSIZE+1, BLOCKSIZE,
         dst, dst_pitch, src, src_pitch, width, height);

   CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
}

void fillin(CuMatrixBase<BaseFloat> &dest, vector< CuMatrix<BaseFloat> > &src, int stream_num){

   for(int i = 0; i < stream_num; ++i){
      BaseFloat *src_data  = src[i].Data();
      BaseFloat *dest_data = dest.Data() + dest.Stride() * i;
      size_t dst_pitch = dest.Stride() * stream_num;
      size_t src_pitch = src[i].Stride();
      size_t width     = src[i].NumCols();
      size_t height    = src[i].NumRows();

      if(height != 0)
         cuMemCopy(dest_data, dst_pitch, src_data, src_pitch, width, height);
   }

   //check
   //CuMatrix<BaseFloat> tmp(dest.NumRows(), dest.NumCols(), kSetZero);
   //for(int i = 0; i < stream_num; ++i){
   //   for(int j = 0; j < src[i].NumRows(); ++j)
   //      tmp.Row(j*stream_num + i).CopyFromVec(src[i].Row(j));
   //}

   //assert(Same(dest, tmp));
}

void weighted_sum(CuMatrixBase<BaseFloat> &out,
      CuVectorG<BaseFloat*> &in_data_arr, const CuVectorG<int32> &in_stride_arr, 
      const CuMatrixBase<BaseFloat> &att){
   Timer tim;
   
   int rows = out.NumRows();
   int cols = out.NumCols();
   int threads = rows * cols;

   cuda_weighted_sum((threads-1)/BLOCKSIZE+1, BLOCKSIZE,
         out.Data(), rows, cols, out.Stride(), 
         in_data_arr.Data(), in_data_arr.Dim(),
         in_stride_arr.Data(), att.Data(), att.Stride());

   CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
}

void back_weighted_sum(CuVectorG<BaseFloat*> &in_diff_arr, const CuVectorG<int32> &in_diff_stride,
      CuMatrixBase<BaseFloat> &att_diff,
      const CuMatrixBase<BaseFloat> &out_diff,
      CuVectorG<BaseFloat*> &in_arr, const CuVectorG<int32> &in_stride,
      const CuMatrixBase<BaseFloat> &att){

   Timer tim;
   
   int rows = out_diff.NumRows();
   int cols = out_diff.NumCols();
   int threads = rows * cols;

   cuda_back_weighted_sum((threads-1)/BLOCKSIZE+1, BLOCKSIZE,
         in_diff_arr.Data(), in_diff_arr.Dim(), rows, cols, in_diff_stride.Data(),
         out_diff.Data(), out_diff.Stride(),
         att.Data(), att.Stride());

   threads = rows * in_diff_arr.Dim();
   cuda_back_weighted_att((threads-1)/BLOCKSIZE+1, BLOCKSIZE,
         att_diff.Data(), rows, cols, in_diff_arr.Dim(), att_diff.Stride(), 
         out_diff.Data(), out_diff.Stride(), in_arr.Data(), in_stride.Data());

   CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
}

