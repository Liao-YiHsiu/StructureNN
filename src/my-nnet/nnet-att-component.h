#ifndef _NNET_ATT_COMPONENT_
#define _NNET_ATT_COMPONENT_

#include "nnet/nnet-utils.h"
#include "cudamatrix/cu-math.h"
#include "util/text-utils.h"

class AttComponent{
   public:
      AttComponent(){}
      ~AttComponent(){}

      AttComponent* Copy() { return new AttComponent();}

      void Propagate(vector< MyCuMatrix<BaseFloat> > &in_arr,
            const CuMatrixBase<BaseFloat> &att, MyCuMatrix<BaseFloat>* out){
         int rows = in_arr[0].NumRows();
         int cols = in_arr[0].NumCols();

         // <check dimension>
         for(int i = 0; i < in_arr.size(); ++i){
            assert(rows == in_arr[i].NumRows());
            assert(cols == in_arr[i].NumCols());
         }

         assert(att.NumRows() == rows);
         assert(att.NumCols() == in_arr.size());
         // </check dimension>

         out->Resize(rows, cols);

         vector<BaseFloat*> in_arr_data;
         vector<int32>      in_arr_stride;

         getCuData(in_arr, in_arr_data, in_arr_stride);
         in_data_arr_   = in_arr_data;
         in_stride_arr_ = in_arr_stride;

         weighted_sum(*out, in_data_arr_, in_stride_arr_, att);

         // check error
         CuMatrix<BaseFloat> tmp_out(out->NumRows(), out->NumCols());
         Matrix<BaseFloat>   att_host(att.NumRows(), att.NumCols());
         att_host.CopyFromMat(att);
         for(int i = 0; i < tmp_out.NumRows(); ++i){
            for(int j = 0; j < in_arr.size(); ++j){
               tmp_out.Row(i).AddVec(att_host(i, j), in_arr[j].Row(i));
            }
         }

         assert(Same(*out, tmp_out));
      }

      void Backpropagate(vector< MyCuMatrix<BaseFloat> > &in_arr,
            const CuMatrixBase<BaseFloat> &att, const CuMatrixBase<BaseFloat> &out_diff,
            vector< MyCuMatrix<BaseFloat> > &in_diff_arr, MyCuMatrix<BaseFloat> &att_diff){
         int rows = in_arr[0].NumRows();
         int cols = in_arr[0].NumCols();

         // <check dimension>
         for(int i = 0; i < in_arr.size(); ++i){
            assert(rows == in_arr[i].NumRows());
            assert(cols == in_arr[i].NumCols());
         }

         assert(att.NumRows() == rows);
         assert(att.NumCols() == in_arr.size());
         assert(out_diff.NumRows() == rows);
         assert(out_diff.NumCols() == cols);

         assert(in_diff_arr.size() == in_arr.size());
         // </check dimension>
         for(int i = 0; i < in_arr.size(); ++i)
            in_diff_arr[i].Resize(rows, cols, kUndefined);
         att_diff.Resize(att.NumRows(), att.NumCols());

         vector<BaseFloat*> in_arr_data;
         vector<int32>      in_arr_stride;
         getCuData(in_arr, in_arr_data, in_arr_stride);
         in_data_arr_   = in_arr_data;
         in_stride_arr_ = in_arr_stride;

         vector<BaseFloat*> in_diff_arr_data;
         vector<int32>      in_diff_arr_stride;
         getCuData(in_diff_arr, in_diff_arr_data, in_diff_arr_stride);
         in_diff_data_arr_   = in_diff_arr_data;
         in_diff_stride_arr_ = in_diff_arr_stride;

         back_weighted_sum(in_diff_data_arr_, in_diff_stride_arr_, att_diff,
               out_diff, in_data_arr_, in_stride_arr_, att);

         // check in_diff_arr and att_diff
         vector< CuMatrix<BaseFloat> > in_diff_tmp(in_arr.size());
         Matrix<BaseFloat>             att_diff_tmp_host(att.NumRows(), att.NumCols());
         for(int i = 0; i < in_diff_tmp.size(); ++i)
            in_diff_tmp[i].Resize(rows, cols);

         Matrix<BaseFloat> att_host(att.NumRows(), att.NumCols());
         att_host.CopyFromMat(att);

         for(int i = 0; i < in_diff_tmp.size(); ++i){
            for(int j = 0; j < rows; ++j){
               in_diff_tmp[i].Row(i).AddVec(att_host(j, i), out_diff.Row(j));
            }
         }

         CuVector<BaseFloat> tmp_vec;
         for(int i = 0; i < rows; ++i){
            for(int j = 0; j < in_diff_tmp.size(); ++j){
               tmp_vec = out_diff.Row(i);
               tmp_vec.MulElements(in_arr[j].Row(i));
               att_diff_tmp_host(i, j) = tmp_vec.Sum();
            }
         }

         for(int i = 0; i < in_diff_tmp.size(); ++i)
            assert(Same(in_diff_tmp[i], in_diff_arr[i]));

         CuMatrix<BaseFloat> att_diff_tmp(att.NumRows(), att.NumCols());
         att_diff_tmp.CopyFromMat(att_diff_tmp_host);
         assert(Same(att_diff_tmp, att_diff));
      }
   private:
      CuVectorG<BaseFloat*> in_data_arr_;
      CuVectorG<BaseFloat*> in_diff_data_arr_;
      CuVectorG<int32>      in_stride_arr_;
      CuVectorG<int32>      in_diff_stride_arr_;
};

#endif
