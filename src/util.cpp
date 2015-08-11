#include "util.h"
#include "kernel.h"

inline double sigmoid(double x){
   return 1/(1+ exp(-x));
}

double frame_acc(const vector<uchar>& path1, const vector<uchar>& path2, double param){

   assert(path1.size() == path2.size());
   int corr = 0;
   for(int i = 0; i < path1.size(); ++i){
      corr += (path1[i] == path2[i]) ? 1:0;
   }

   return corr / (double)path1.size();
}

// reference is path1.
double phone_acc(const vector<uchar>& path1, const vector<uchar>& path2, double inst){
   assert(path1.size() == path2.size());

   vector<uchar> path1_trim;
   vector<uchar> path2_trim;
   trim_path(path1, path1_trim);
   trim_path(path2, path2_trim);

   vector<int32> path1_trim_32;
   vector<int32> path2_trim_32;

   UcharToInt32(path1_trim, path1_trim_32);
   UcharToInt32(path2_trim, path2_trim_32);

   int in, de, su;

   int32 dist = LevenshteinEditDistance(path1_trim_32, path2_trim_32, &in, &de, &su);

   double corr = path1_trim.size() - (in*inst + de + su); 

   if(corr < 0) corr = 0;

   return corr / (double)path1_trim.size();
}

void UcharToInt32(const vector<uchar>& src_path, vector<int32>& des_path){
   des_path.resize(src_path.size());
   for(int i = 0; i < src_path.size(); ++i)
      des_path[i] = src_path[i];
}

void Int32ToUchar(const vector<int32>& src_path, vector<uchar>& des_path){
   des_path.resize(src_path.size());
   for(int i = 0; i < src_path.size(); ++i){
      des_path[i] = src_path[i];
      assert(src_path[i] == des_path[i]);
   }
}


//void makeFeature(const Matrix<BaseFloat> &feat, const vector<uchar> &path, int32 maxState, SubVector<BaseFloat> vec){
//   assert(feat.NumRows() == path.size());
//
//   int feat_dim = feat.NumCols();
//
//   SubVector<BaseFloat> tran(vec, feat_dim * maxState, maxState*maxState);
//   for(int i = 0; i < path.size(); ++i){
//      SubVector<BaseFloat> obs(vec, (path[i]-1)*feat_dim, feat_dim);
//      //int32 offset = (path[i]-1)*feat_dim;
//      //for(int k = 0; k < feat_dim; ++k)
//      //   vec(offset+k) = feat(i,k); 
//      obs.AddVec(1, feat.Row(i));
//
//      if(i > 0){
//         tran((path[i-1]-1)*maxState + path[i]-1) += 1;
//      }
//   }
//
//   // normalization
//   vec.Scale(1/(double)path.size());
//}
//
//void* makeFeatureP(void *param){
//   FData* fData = (FData*) param;
//
//   for(int i = 0; i < fData->maxState; ++i){
//      vector<uchar> path = *(fData->path);
//      path[fData->chgID] = i+1;
//      makeFeature(*(fData->feat), path, fData->maxState, fData->mat->Row(i));
//   }
//   return NULL;
//}
//
//void makeFeatureBatch(const Matrix<BaseFloat> &feat, const vector<int32> &path, int chgID, int32 maxState, SubMatrix<BaseFloat> mat){
//   assert(feat.NumRows() == path.size());
//   assert(mat.NumRows() == maxState);
//
//   assert(false);
//  //  int feat_dim = feat.NumCols();
//
//  //  // compute commonly used vector
//  //  {
//  //     SubVector<BaseFloat> vec = mat.Row(0);
//
//  //     SubVector<BaseFloat> tran(vec, feat_dim * maxState, maxState*maxState);
//  //     for(int i = 0; i < path.size(); ++i){
//  //        if(i == chgID)continue;
//
//  //        SubVector<BaseFloat> obs(vec, (path[i]-1)*feat_dim, feat_dim);
//  //        obs.AddVec(1, feat.Row(i));
//
//  //        if(i > 0 && i-1 != chgID){
//  //           tran((path[i-1]-1)*maxState + path[i]-1) += 1;
//  //        }
//  //     }
//  //     // copy to specified Matrix
//  //     for(int i = 1; i < maxState; ++i)
//  //        mat.Row(i).CopyFromVec(vec);
//  //  }
//
//
//  //  for(int i = 0; i < maxState; ++i){
//  //     SubVector<BaseFloat> vec = mat.Row(i);
//
//  //     SubVector<BaseFloat> tran(vec, feat_dim * maxState, maxState*maxState);
//  //     SubVector<BaseFloat> obs(vec, i*feat_dim, feat_dim);
//  //     obs.AddVec(1, feat.Row(chgID));
//
//  //     if(chgID >= 1)
//  //        tran((path[chgID-1]-1)*maxState + i) += 1;
//  //     if(chgID+1 < path.size())
//  //        tran(i*maxState + path[chgID+1]-1) += 1;
//  //  }
//
//  //  // normalization
//  //  for(int i = 1; i < maxState; ++i)
//  //        mat.Scale(1/(double)path.size());
//
//}

//void makeFeature(const CuMatrix<BaseFloat> &feat, const vector<int32> &path, int32 maxState, CuSubVector<BaseFloat> vec){
//   assert(feat.NumRows() == path.size());
//
//   int feat_dim = feat.NumCols();
//
//   Vector<BaseFloat> tran_tmp(maxState*maxState);
//   for(int i = 0; i < path.size(); ++i){
//      CuSubVector<BaseFloat> obs(vec, (path[i]-1)*feat_dim, feat_dim);
//      //int32 offset = (path[i]-1)*feat_dim;
//      //for(int k = 0; k < feat_dim; ++k)
//      //   vec(offset+k) = feat(i,k); 
//      obs.AddVec(1, feat.Row(i));
//
//      if(i > 0){
//         tran_tmp((path[i-1]-1)*maxState + path[i]-1) += 1;
//      }
//   }
//
//   CuSubVector<BaseFloat> tran(vec, feat_dim * maxState, maxState*maxState);
//   tran.CopyFromVec(tran_tmp);
//
//   // normalization
//   vec.Scale(1/(double)path.size());
//}

/*
void makePost(const vector<int32> &realPath, const vector<int32> &path, Posterior &post){
   double acc = path_acc(realPath, path);

   vector< pair<int32, BaseFloat> > arr; 

   if(acc != 0.0)
      arr.push_back(make_pair(0, acc));
   if(acc != 1.0)
      arr.push_back(make_pair(1, 1-acc));

   post.push_back(arr);
}
*/

//void makePost(double acc, Posterior &post){
//   vector< pair<int32, BaseFloat> > arr; 
//
//   if(acc != 0.0)
//      arr.push_back(make_pair(0, acc));
//   if(acc != 1.0)
//      arr.push_back(make_pair(1, 1-acc));
//
//   post.push_back(arr);
//}
//
//
//int32 sample(const vector<BaseFloat> &arr){
//   BaseFloat sum = 0;
//   for(int i = 0; i < arr.size(); ++i)
//      sum += arr[i];
//   BaseFloat p = rand() / (double) RAND_MAX * sum;
//   sum = 0;
//   for(int i = 0; i < arr.size(); ++i){
//      sum += arr[i];
//      if(sum >= p ) return i;
//   }
//   assert(false);
//   return -1;
//}

int32 best(const vector<BaseFloat> &arr){
   assert(arr.size() >= 1);
   BaseFloat max = arr[0];
   int32 index = 0;
   for(int i = 1; i < arr.size(); ++i)
     if(max < arr[i]){
        max = arr[i];
        index = i;
     }
   return index;
}

void trim_path(const vector<uchar>& scr_path, vector<uchar>& des_path){
   des_path.clear();

   int32 prev = scr_path[0];
   des_path.push_back(scr_path[0]);

   for(int i = 1; i < scr_path.size(); ++i){
      if(prev != scr_path[i]){
         prev = scr_path[i];
         des_path.push_back(scr_path[i]);
      }
   }
}

// void findMax(const myCuVector<BaseFloat> &arr, Vector<BaseFloat> &arr_host, int &index_host, BaseFloat &max_host){
//    assert(false);
//  //   arr_host.Resize(arr.Dim());
//  //   arr_host.CopyFromVec(arr);
// 
//  //   max_host = arr_host(0);
//  //   index_host = 0;
//  //   for(int i = 1; i < arr_host.Dim(); ++i){
//  //      if(arr_host(i) >= max_host){
//  //         max_host = arr_host(i);
//  //         index_host = i;
//  //      }
//  //   }
// }
// 
// // to be removed
// bool updateLabelCuda(const myCuVector<BaseFloat> &arr, int row, CuIntVector &lab, int l, int S, BaseFloat &value){
//    assert(false);
//  //   
//  //   Timer tim;
//  //   // find max prob
//  //   int sharemem = BLOCKSIZE*(sizeof(int) + sizeof(BaseFloat));
//  //
//  //   assert(S < BLOCKSIZE);
//  //
//  //   myCuVector<BaseFloat> tmparr(1);
//  //   CuIntVector           tmpidx(1);
//  //
//  //   cuda_find_max(1, BLOCKSIZE, sharemem, arr.Data()+ row*S , S, tmparr.Data(), tmpidx.Data());
//  //
//  //   Vector<BaseFloat> host_arr(tmparr.Dim());
//  //   vector<int32>     host_idx(tmpidx.Dim());
//  //   vector<int32>     host_lab(lab.Dim());
//  //
//  //   host_arr.CopyFromVec(tmparr);
//  //   tmpidx.CopyToVec(host_idx);
//  //   lab.CopyToVec(host_lab);
//  //
//  //   value = host_arr(0);
//  //
//  //   int32 s = host_idx[0];
//  //
//  //   CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
//  //
//  //   assert(l < host_lab.size());
//  //
//  //   if(host_lab[l] == s)
//  //      return false;
//  //
//  //   // update
//  //   host_lab[l] = s;
//  //   lab.CopyFromVec(host_lab);
//  //
//    return true;
// }
// 
// bool updateLabelCuda(const myCuVector<BaseFloat> &arr, CuIntVector &lab, int S, BaseFloat &value){
//    Timer tim;
//    assert(false);
// 
//  //   // find max prob
//  //   int blocks = arr.Dim()/BLOCKSIZE + 1;
//  //   int sharemem = BLOCKSIZE*(sizeof(int) + sizeof(BaseFloat));
//  //   assert(blocks < BLOCKSIZE);
//  //
//  //
//  //   myCuVector<BaseFloat> tmparr(blocks + 1);
//  //   CuIntVector           tmpidx(blocks + 1);
//  //
//  //   cuda_find_max(blocks, BLOCKSIZE, sharemem, arr.Data(), arr.Dim(), (BaseFloat*)tmparr.Data()+1, (int*)tmpidx.Data()+1);
//  //
//  //   cuda_find_max(1, BLOCKSIZE, sharemem, tmparr.Data() + 1, blocks, tmparr.Data(), tmpidx.Data()); 
//  //   // update lable
//  //
//  //   Vector<BaseFloat> host_arr(tmparr.Dim());
//  //   vector<int32>     host_idx(tmpidx.Dim());
//  //   vector<int32>     host_lab(lab.Dim());
//  //
//  //   host_arr.CopyFromVec(tmparr);
//  //   tmpidx.CopyToVec(host_idx);
//  //   lab.CopyToVec(host_lab);
//  //
//  //   value = host_arr(0);
//  //
//  //   int32 index = host_idx[host_idx[0]+1];
//  //
//  //   int32 l = index/S;
//  //   int32 s = index%S;
//  //
//  //   CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
//  //
//  //   assert(l < host_lab.size());
//  //
//  //   if(host_lab[l] == s)
//  //      return false;
//  //
//  //   // update
//  //   host_lab[l] = s;
//  //   lab.CopyFromVec(host_lab);
//  //
//    return true;
// }
// 
// void makeFeatureCuda(const myCuMatrix<BaseFloat> &feats, const CuIntVector &lab, int l, int S, myCuMatrix<BaseFloat> &ret, int ret_row){
//    Timer tim;
// 
//    MatrixDim dim = feats.Dim();
//    int L = dim.rows;
//    int F = dim.cols;
// 
//    assert( L == lab.Dim() );
// 
//    //KALDI_LOG << "sum = " << ret.Sum();
// 
//    assert( cudaSuccess == cudaGetLastError() );
//    cuda_make_obs( (S * F)/BLOCKSIZE+1, BLOCKSIZE, feats.Data(), dim.rows, dim.cols,
//          dim.stride, lab.Data(), l, ret.Data() + ret_row *S* ret.Dim().stride, ret.Dim().stride, S);
// 
//    //KALDI_LOG << "sum = " << ret.Sum();
// 
//    cuda_make_tran( S/BLOCKSIZE+1, BLOCKSIZE, dim.rows, dim.cols, lab.Data(), l,
//          ret.Data() + ret_row *S * ret.Dim().stride, ret.Dim().stride, S);
// 
//    //KALDI_LOG << "sum = " << ret.Sum();
// 
//    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
// 
//    //KALDI_LOG << "sum = " << ret.Sum();
// 
//    // check generated features.
//    //Matrix<BaseFloat> feats_host(feats.NumRows(), feats.NumCols());
//    //feats_host.CopyFromMat(feats);
// 
//    //KALDI_LOG << "feats sum = " << feats_host.Sum();
// 
//    //Matrix<BaseFloat> ret_host(ret.NumRows(), ret.NumCols());
//    //ret_host.CopyFromMat(ret);
//    //
//    //KALDI_LOG << "sum = " << ret_host.Sum();
// 
// 
//    //vector<int> lab_host;
//    //lab.CopyToVec(lab_host);
//    //for(int i = 0; i < lab_host.size(); ++i)
//    //   lab_host[i] += 1;
// 
//    //Matrix<BaseFloat> cpu(S, F*S + S*S);
//    //for(int s = 0; s < S; ++s){
//    //   lab_host[l] = s + 1;
//    //   makeFeature(feats_host, lab_host, S, cpu.Row(s));
//    //}
// 
//    //KALDI_LOG << "sum = " << cpu.Sum();
// 
// 
//    //float err = 0;
//    //for(int i = 0; i < S; ++i)
//    //   for(int j = 0; j < cpu.NumCols(); ++j)
//    //      err += (cpu(i, j) - ret_host(i+ret_row*S, j))*(cpu(i, j) - ret_host(i+ret_row*S, j));
// 
//    //assert(err < 0.01);
// }
// 
// void makeFeatureCuda(const myCuMatrix<BaseFloat> &feats, const CuIntVector &lab, int S, myCuMatrix<BaseFloat> &ret){
//    Timer tim;
//    assert(false);
// 
//  //  MatrixDim dim = feats.Dim();
//  //  int L = dim.rows;
//  //  int F = dim.cols;
// 
//  //  assert( L == lab.Dim() );
// 
//  //  ret.Resize(L*S, F*S + S*S);
// 
//  //  cuda_make_obs( (L * S * F)/BLOCKSIZE+1, BLOCKSIZE, feats.Data(), dim.rows, dim.cols,
//  //        dim.stride, lab.Data(), ret.Data(), ret.Dim().stride, S);
// 
//  //  cuda_make_tran( (L * S)/BLOCKSIZE+1, BLOCKSIZE, dim.rows, dim.cols, lab.Data(), 
//  //        ret.Data(), ret.Dim().stride, S);
//  //  assert( cudaSuccess == cudaGetLastError() );
// 
//  //  CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
// 
// }

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

void CuIntVector::Destroy(){
   if(data_){
      CuDevice::Instantiate().Free(this->data_);
   }
   data_ = 0; dim_ = 0;
}

void CuIntVector::CopyFromVec(const vector<int> &src){
   if(src.size() != dim_)
      Resize(src.size());
   //assert(src.size() == dim_); 
   Timer tim;
   CU_SAFE_CALL(cudaMemcpy(data_, src.data(), dim_ * sizeof(int), cudaMemcpyHostToDevice));
   CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
}

void CuIntVector::CopyToVec(vector<int> &des)const{
   if(des.size() != dim_ ) des.resize(dim_);

   Timer tim;
   CU_SAFE_CALL(cudaMemcpy(des.data(), data_, dim_ * sizeof(int), cudaMemcpyDeviceToHost));
   CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
}

void CuIntVector::Resize(int dim){
   if( dim_ == dim ) return; // don't set zeros

   Destroy();
   dim_ = dim;
   if(dim == 0) return;

   Timer tim;

   this->data_ = static_cast<int*>(CuDevice::Instantiate().Malloc(dim * sizeof(int)));

   CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());    
}

int Strt::Eval(const VectorBase<BaseFloat> &delta, const CuMatrixBase<BaseFloat> &nnet_out, 
      vector<CuMatrix<BaseFloat> > *diff, int* counter, int raw, const vector<int> *example_type){

   int N = delta.Dim();
   int retIdx = -1;
   double maxError = -1;

   KALDI_ASSERT(nnet_out.NumCols() == 1);
   KALDI_ASSERT(N ==  nnet_out.NumRows());

   // copy data from gpu to cpu
   nnet_out_host_.Resize(nnet_out.NumRows(), nnet_out.NumCols(), kUndefined);
   nnet_out_host_.CopyFromMat(nnet_out);

   if(diff != NULL){
      diff_host_[0].Resize(N, 1, kSetZero);
      diff_host_[1].Resize(N, 1, kSetZero);
   }

   double total_error = 0, total_correct = 0;
   for(int i = 0; i < N; ++i){
      double error = nnet_out_host_(i, 0) + delta(i);
      error -= (raw >= 0) ? nnet_out_host_(raw, 0) : 0 ;
      if( error > 0 ){
         if(diff != NULL){
            diff_host_[0](i, 0) = 1;
            diff_host_[1](i, 0) = -1;
         }

         if(error > maxError){
            retIdx = i;
            maxError = error;
         }
         
         if(example_type != NULL)
            loss_arr_[(*example_type)[i]] += error;

         total_error += error;
      }else{
         if(example_type != NULL)
            correct_arr_[(*example_type)[i]] += 1;

         total_correct += 1;
      }

      if(example_type != NULL)
         frames_arr_[(*example_type)[i]] += 1;
   }

   KALDI_ASSERT(KALDI_ISFINITE(total_error));
   if(counter != NULL) *counter = N - total_correct;

   if(diff != NULL){
      KALDI_ASSERT(diff -> size() == 2);
      (*diff)[0] = diff_host_[0];
      (*diff)[1] = diff_host_[1];
   }

   frames_arr_[ALL_TYPE]  += N;
   loss_arr_[ALL_TYPE]    += total_error;
   correct_arr_[ALL_TYPE] += total_correct;

   // progress losss reporting
   {
      static const int32 progress_step = 1024; 
      frames_progress_  += N;
      loss_progress_    += total_error; 
      correct_progress_ += total_correct;

      if (frames_progress_ > progress_step) {
         KALDI_VLOG(1) << "ProgressLoss[last " 
            << static_cast<int>(frames_progress_/progress_step) << "k of " 
            << static_cast<int>(frames_arr_[ALL_TYPE]/progress_step) << "k]: " 
            << loss_progress_/frames_progress_ << " (Strt) " 
            << "FRAME ACC >> " << 100*correct_progress_/frames_progress_ << "% <<";
         // store
         loss_vec_.push_back(loss_progress_/frames_progress_);
         // reset
         frames_progress_  = 0;
         loss_progress_    = 0;
         correct_progress_ = 0;
      }
   }

   return retIdx;

}

string Strt::Report() {
   ostringstream oss;
   oss << "AvgLoss: " << loss_arr_[ALL_TYPE]/frames_arr_[ALL_TYPE] << " (Strt) " << endl;
   for(int i = ALL_TYPE + 1; i < END_TYPE; ++i){
      if(frames_arr_[i] > 0)
         oss << "  " << LABEL_NAME[i] << " Loss: " << loss_arr_[i] / frames_arr_[i] << " (Strt) " << endl;
   }
      
   if (loss_vec_.size() > 0) {
      oss << "progress: [";
      copy(loss_vec_.begin(),loss_vec_.end(),ostream_iterator<float>(oss," "));
      oss << "]" << endl;
   }
   if (correct_arr_[ALL_TYPE] >= 0.0) {
      oss << "\nFRAME_ACCURACY >> " << 100.0*correct_arr_[ALL_TYPE]/frames_arr_[ALL_TYPE] << "% <<" << endl;
      for(int i = ALL_TYPE + 1; i < END_TYPE; ++i){
         if(frames_arr_[i] > 0)
            oss << "  " << LABEL_NAME[i] << " ACC >> " << 
               100.0*correct_arr_[i] / frames_arr_[i] << "% << " << endl;
      }
   }
   return oss.str(); 
}

int StrtCmp::Eval(const VectorBase<BaseFloat> &delta, const CuMatrixBase<BaseFloat> &nnet_out, 
      vector<CuMatrix<BaseFloat> > *diff, int* counter, const vector<int> *example_type){

   int N = delta.Dim();
   int retIdx = -1;
   double maxError = -1;

   KALDI_ASSERT(nnet_out.NumCols() == 1);
   KALDI_ASSERT(N ==  nnet_out.NumRows());

   // copy data from gpu to cpu
   nnet_out_host_.Resize(nnet_out.NumRows(), nnet_out.NumCols(), kUndefined);
   nnet_out_host_.CopyFromMat(nnet_out);

   if(diff != NULL){
      diff_host_[0].Resize(N, 1, kSetZero);
      diff_host_[1].Resize(N, 1, kSetZero);
   }

   double total_error = 0, total_correct = 0;
   for(int i = 0; i < N; ++i){
      assert(delta(i) != 0);
      int sign_delta = delta(i) > 0 ? 1: -1;
      double error = exp(-sign_delta * nnet_out_host_(i, 0));

      if(error > 1e4) error = 1e4;

      if(diff != NULL){
         diff_host_[0](i, 0) = -sign_delta * error;
         diff_host_[1](i, 0) =  sign_delta * error;
      }

      if( -sign_delta * nnet_out_host_(i, 0) < 0 ){
         if(example_type != NULL)
            correct_arr_[(*example_type)[i]] += 1;

         total_correct += 1;
      }

      if(example_type != NULL)
         loss_arr_[(*example_type)[i]] += error;

      total_error += error;

      if(example_type != NULL)
         frames_arr_[(*example_type)[i]] += 1;
   }

   KALDI_ASSERT(KALDI_ISFINITE(total_error));
   if(counter != NULL) *counter = N - total_correct;

   if(diff != NULL){
      KALDI_ASSERT(diff -> size() == 2);
      (*diff)[0] = diff_host_[0];
      (*diff)[1] = diff_host_[1];
   }

   frames_arr_[ALL_TYPE]  += N;
   loss_arr_[ALL_TYPE]    += total_error;
   correct_arr_[ALL_TYPE] += total_correct;

   // progress losss reporting
   {
      static const int32 progress_step = 3600; 
      frames_progress_  += N;
      loss_progress_    += total_error; 
      correct_progress_ += total_correct;

      if (frames_progress_ > progress_step) {
         KALDI_VLOG(1) << "ProgressLoss[last " 
            << static_cast<int>(frames_progress_/progress_step) << "h of " 
            << static_cast<int>(frames_arr_[ALL_TYPE]/progress_step) << "h]: " 
            << loss_progress_/frames_progress_ << " (StrtCmp) " 
            << "FRAME ACC >> " << 100*correct_progress_/frames_progress_ << "% <<";
         // store
         loss_vec_.push_back(loss_progress_/frames_progress_);
         // reset
         frames_progress_  = 0;
         loss_progress_    = 0;
         correct_progress_ = 0;
      }
   }

   return retIdx;

}

string StrtCmp::Report() {
   ostringstream oss;
   oss << "AvgLoss: " << loss_arr_[ALL_TYPE]/frames_arr_[ALL_TYPE] << " (StrtCmp) " << endl;
   for(int i = ALL_TYPE + 1; i < END_TYPE*END_TYPE; ++i){
      if(frames_arr_[i] > 0){
         int m = i / END_TYPE;
         int n = i % END_TYPE;
         oss << "  " << LABEL_NAME[m] << " <-> " << LABEL_NAME[n]
            << " Loss: " << loss_arr_[i] / frames_arr_[i] << " (StrtCmp) " << endl;
      }
   }
      
   if (loss_vec_.size() > 0) {
      oss << "progress: [";
      copy(loss_vec_.begin(),loss_vec_.end(),ostream_iterator<float>(oss," "));
      oss << "]" << endl;
   }
   if (correct_arr_[ALL_TYPE] >= 0.0) {
      oss << "\nFRAME_ACCURACY >> " << 100.0*correct_arr_[ALL_TYPE]/frames_arr_[ALL_TYPE] << "% <<" << endl;
      for(int i = ALL_TYPE + 1; i < END_TYPE*END_TYPE; ++i){
         if(frames_arr_[i] > 0){
            int m = i / END_TYPE;
            int n = i % END_TYPE;
            oss << "  " << LABEL_NAME[m] << " <-> " << LABEL_NAME[n] << " ACC >> " << 
               100.0*correct_arr_[i] / frames_arr_[i] << "% << " << endl;
         }
      }
   }
   return oss.str(); 
}

// use second value as key
void readPhMap(const string path, const string id_path, map<string, int> &phMap){
   map<string, int> inner;
   string line, tmp;
   int32 id;

   {
      ifstream fin(id_path.c_str());
      while(getline(fin, line)){
         stringstream ss(line);
         ss >> tmp >> id;
         inner[tmp] = id;
         assert(id == inner[tmp]);
      }

   }

   ifstream fin(path.c_str());

   while(getline(fin, line)){
      stringstream ss(line);
      string tmp2;
      ss >> tmp >> tmp2;

      if(tmp2.empty()
            || inner.find(tmp2) == inner.end()){

         phMap[tmp] = -1;
         continue;
      }

      phMap[tmp] = inner[tmp2];
   }
}

void getPhone(const string &key, const string &timit, map<string, int> &phMap, vector<uchar> &phIdx){

   // get phone path
   size_t pos = key.find('_');
   string spk = key.substr(0, pos);
   string sen = key.substr(pos+1);

   char buf[BUFSIZE];
   sprintf(buf, "ls %s/*/*/%s/%s.phn", timit.c_str(), spk.c_str(), sen.c_str());
   string path = execute(buf); 

   //analyze phone
   ifstream fin(path.substr(0, path.find('\n')).c_str());
   int s, e;
   string phn;

   vector<int> ph_e;
   vector<int> ph_idx;

   while(fin >> s >> e >> phn){
      if(phMap.find(phn) == phMap.end() || phMap[phn] < 0){
         int size = ph_e.size();
         if(size == 0)
            continue;
         else
            ph_e[size-1] = (ph_e[size-1] + e)/2;
      }else{
         ph_e.push_back(e);
         ph_idx.push_back(phMap[phn]);
      }
   }
   ph_e[ph_e.size()-1] = e;

   double step = ph_e[ph_e.size()-1] /(double) phIdx.size();
   int j = 0;
   for(int i = 0, size = phIdx.size(); i < size; ++i){
      double now = step/2 + i*step;
      while(now >= ph_e[j]) j++;
      phIdx[i] = ph_idx[j];
   }
}

string execute(const string &cmd){
   FILE* pipe = popen(cmd.c_str(), "r");
   assert(pipe);

   string ret;
   char buf[BUFSIZE];
   while(!feof(pipe)){
      if( fgets(buf, BUFSIZE, pipe) != NULL)
         ret.append(buf);
   }
   pclose(pipe);
   return ret;
}

void print(const MatrixBase<BaseFloat> &mat, int row){
   BaseFloat sum = 0;
   BaseFloat sqr_sum = 0;
   int counter = 0;


   int start = 0, end = mat.NumRows();
   if(row >= 0){
      start = row; end = row+1;
   }

   for(int i = start; i < end; ++i){
      for(int j = 0; j < mat.NumCols(); ++j){
         cout << mat(i, j) << " ";
         sum += mat(i, j);
         sqr_sum += mat(i,j) * mat(i, j);
         counter++;
      }
      cout << endl;
   }
   cout << "Sum: " << sum << "\t SSUM: " << sqr_sum << "\t CNT: " << counter<< endl;
}

void print(const CuMatrixBase<BaseFloat> &cumat, int row){
   Matrix<BaseFloat> tmp(cumat.NumRows(), cumat.NumCols());
   tmp.CopyFromMat(cumat);
   print(tmp, row);
}
