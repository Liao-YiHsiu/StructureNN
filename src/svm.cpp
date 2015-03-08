#include "svm.h"
#include "kernel.h"



double frame_acc(const vector<int32>& path1, const vector<int32>& path2){

   assert(path1.size() == path2.size());
   int corr = 0;
   for(int i = 0; i < path1.size(); ++i){
      corr += (path1[i] == path2[i]) ? 1:0;
   }

   return corr / (double)path1.size();
}

// reference is path1.
double phone_acc(const vector<int32>& path1, const vector<int32>& path2){
   assert(path1.size() == path2.size());

   vector<int32> path1_trim;
   vector<int32> path2_trim;
   trim_path(path1, path1_trim);
   trim_path(path2, path2_trim);

   int32 dist = LevenshteinEditDistance(path1_trim, path2_trim);

   int32 corr = path1_trim.size() - dist; 

   if(corr < 0) corr = 0;

   return corr / (double)path1_trim.size();
}



void makeFeature(const Matrix<BaseFloat> &feat, const vector<int32> &path, int32 maxState, SubVector<BaseFloat> vec){
   assert(feat.NumRows() == path.size());

   int feat_dim = feat.NumCols();

   SubVector<BaseFloat> tran(vec, feat_dim * maxState, maxState*maxState);
   for(int i = 0; i < path.size(); ++i){
      SubVector<BaseFloat> obs(vec, (path[i]-1)*feat_dim, feat_dim);
      //int32 offset = (path[i]-1)*feat_dim;
      //for(int k = 0; k < feat_dim; ++k)
      //   vec(offset+k) = feat(i,k); 
      obs.AddVec(1, feat.Row(i));

      if(i > 0){
         tran((path[i-1]-1)*maxState + path[i]-1) += 1;
      }
   }

   // normalization
   vec.Scale(1/(double)path.size());
}

void* makeFeatureP(void *param){
   FData* fData = (FData*) param;

   for(int i = 0; i < fData->maxState; ++i){
      vector<int32> path = *(fData->path);
      path[fData->chgID] = i+1;
      makeFeature(*(fData->feat), path, fData->maxState, fData->mat->Row(i));
   }
   return NULL;
}

void makeFeatureBatch(const Matrix<BaseFloat> &feat, const vector<int32> &path, int chgID, int32 maxState, SubMatrix<BaseFloat> mat){
   assert(feat.NumRows() == path.size());
   assert(mat.NumRows() == maxState);

   int feat_dim = feat.NumCols();

   // compute commonly used vector
   {
      SubVector<BaseFloat> vec = mat.Row(0);

      SubVector<BaseFloat> tran(vec, feat_dim * maxState, maxState*maxState);
      for(int i = 0; i < path.size(); ++i){
         if(i == chgID)continue;

         SubVector<BaseFloat> obs(vec, (path[i]-1)*feat_dim, feat_dim);
         obs.AddVec(1, feat.Row(i));

         if(i > 0 && i-1 != chgID){
            tran((path[i-1]-1)*maxState + path[i]-1) += 1;
         }
      }
      // copy to specified Matrix
      for(int i = 1; i < maxState; ++i)
         mat.Row(i).CopyFromVec(vec);
   }


   for(int i = 0; i < maxState; ++i){
      SubVector<BaseFloat> vec = mat.Row(i);

      SubVector<BaseFloat> tran(vec, feat_dim * maxState, maxState*maxState);
      SubVector<BaseFloat> obs(vec, i*feat_dim, feat_dim);
      obs.AddVec(1, feat.Row(chgID));

      if(chgID >= 1)
         tran((path[chgID-1]-1)*maxState + i) += 1;
      if(chgID+1 < path.size())
         tran(i*maxState + path[chgID+1]-1) += 1;
   }

   // normalization
   for(int i = 1; i < maxState; ++i)
         mat.Scale(1/(double)path.size());

}

void makeFeature(const CuMatrix<BaseFloat> &feat, const vector<int32> &path, int32 maxState, CuSubVector<BaseFloat> vec){
   assert(feat.NumRows() == path.size());

   int feat_dim = feat.NumCols();

   Vector<BaseFloat> tran_tmp(maxState*maxState);
   for(int i = 0; i < path.size(); ++i){
      CuSubVector<BaseFloat> obs(vec, (path[i]-1)*feat_dim, feat_dim);
      //int32 offset = (path[i]-1)*feat_dim;
      //for(int k = 0; k < feat_dim; ++k)
      //   vec(offset+k) = feat(i,k); 
      obs.AddVec(1, feat.Row(i));

      if(i > 0){
         tran_tmp((path[i-1]-1)*maxState + path[i]-1) += 1;
      }
   }

   CuSubVector<BaseFloat> tran(vec, feat_dim * maxState, maxState*maxState);
   tran.CopyFromVec(tran_tmp);

   // normalization
   vec.Scale(1/(double)path.size());
}

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

void makePost(double acc, Posterior &post){
   vector< pair<int32, BaseFloat> > arr; 

   if(acc != 0.0)
      arr.push_back(make_pair(0, acc));
   if(acc != 1.0)
      arr.push_back(make_pair(1, 1-acc));

   post.push_back(arr);
}


int32 sample(const vector<BaseFloat> &arr){
   BaseFloat sum = 0;
   for(int i = 0; i < arr.size(); ++i)
      sum += arr[i];
   BaseFloat p = rand() / (double) RAND_MAX * sum;
   sum = 0;
   for(int i = 0; i < arr.size(); ++i){
      sum += arr[i];
      if(sum >= p ) return i;
   }
   assert(false);
   return -1;
}

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

void trim_path(const vector<int32>& scr_path, vector<int32>& des_path){
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

void findMax(const myCuVector<BaseFloat> &arr, Vector<BaseFloat> &arr_host, int &index_host, BaseFloat &max_host){
   arr_host.Resize(arr.Dim());
   arr_host.CopyFromVec(arr);

   max_host = arr_host(0);
   index_host = 0;
   for(int i = 1; i < arr_host.Dim(); ++i){
      if(arr_host(i) >= max_host){
         max_host = arr_host(i);
         index_host = i;
      }
   }
}

bool updateLabelCuda(const myCuVector<BaseFloat> &arr, int row, CuIntVector &lab, int l, int S, BaseFloat &value){
   Timer tim;
   // find max prob
   int sharemem = BLOCKSIZE*(sizeof(int) + sizeof(BaseFloat));

   assert(S < BLOCKSIZE);

   myCuVector<BaseFloat> tmparr(1);
   CuIntVector           tmpidx(1);

   cuda_find_max(1, BLOCKSIZE, sharemem, arr.Data()+ row*S , S, tmparr.Data(), tmpidx.Data());

   Vector<BaseFloat> host_arr(tmparr.Dim());
   vector<int32>     host_idx(tmpidx.Dim());
   vector<int32>     host_lab(lab.Dim());

   host_arr.CopyFromVec(tmparr);
   tmpidx.CopyToVec(host_idx);
   lab.CopyToVec(host_lab);

   value = host_arr(0);

   int32 s = host_idx[0];

   CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());

   assert(l < host_lab.size());

   if(host_lab[l] == s)
      return false;

   // update
   host_lab[l] = s;
   lab.CopyFromVec(host_lab);

   return true;
   //_update_label<<<1, 1>>>(tmparr.Data(), tmpidx.Data(), lab.Data(), lab.Dim(), S);
   //val = tmparr(0);
}

bool updateLabelCuda(const myCuVector<BaseFloat> &arr, CuIntVector &lab, int S, BaseFloat &value){
   Timer tim;

   // use CPU to find
   //BaseFloat max_host;
   //int index_host;
   //Vector<BaseFloat> arr_host;
   //findMax(arr, arr_host, index_host, max_host);


   // find max prob
   int blocks = arr.Dim()/BLOCKSIZE + 1;
   int sharemem = BLOCKSIZE*(sizeof(int) + sizeof(BaseFloat));
   assert(blocks < BLOCKSIZE);


   myCuVector<BaseFloat> tmparr(blocks + 1);
   CuIntVector           tmpidx(blocks + 1);

   //KALDI_LOG << "sum = " << tmparr.Sum();

   //assert( cudaSuccess == cudaGetLastError() );

   cuda_find_max(blocks, BLOCKSIZE, sharemem, arr.Data(), arr.Dim(), (BaseFloat*)tmparr.Data()+1, (int*)tmpidx.Data()+1);
   //assert( cudaSuccess == cudaGetLastError() );

   //BaseFloat tmpmax_host;
   //int tmpindex_host;
   //Vector<BaseFloat> tmparr_host;
   //findMax(tmparr, tmparr_host, tmpindex_host, tmpmax_host);

   //KALDI_LOG << "sum = " << tmparr.Sum();

   cuda_find_max(1, BLOCKSIZE, sharemem, tmparr.Data() + 1, blocks, tmparr.Data(), tmpidx.Data()); 
   //assert( cudaSuccess == cudaGetLastError() );

   //KALDI_LOG << "sum = " << tmparr.Sum();

   // update lable

   Vector<BaseFloat> host_arr(tmparr.Dim());
   vector<int32>     host_idx(tmpidx.Dim());
   vector<int32>     host_lab(lab.Dim());

   //KALDI_LOG << "sum = " << tmparr.Sum();

   host_arr.CopyFromVec(tmparr);
   tmpidx.CopyToVec(host_idx);
   lab.CopyToVec(host_lab);

   value = host_arr(0);

   int32 index = host_idx[host_idx[0]+1];

   //assert(index == index_host || arr_host(index) == arr_host(index_host));

   int32 l = index/S;
   int32 s = index%S;

   CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());

   assert(l < host_lab.size());

   if(host_lab[l] == s)
      return false;

   // update
   host_lab[l] = s;
   lab.CopyFromVec(host_lab);

   return true;
   //_update_label<<<1, 1>>>(tmparr.Data(), tmpidx.Data(), lab.Data(), lab.Dim(), S);
   //val = tmparr(0);
}

void makeFeatureCuda(const myCuMatrix<BaseFloat> &feats, const CuIntVector &lab, int l, int S, myCuMatrix<BaseFloat> &ret, int ret_row){
   Timer tim;

   MatrixDim dim = feats.Dim();
   int L = dim.rows;
   int F = dim.cols;

   assert( L == lab.Dim() );

   //KALDI_LOG << "sum = " << ret.Sum();

   assert( cudaSuccess == cudaGetLastError() );
   cuda_make_obs( (S * F)/BLOCKSIZE+1, BLOCKSIZE, feats.Data(), dim.rows, dim.cols,
         dim.stride, lab.Data(), l, ret.Data() + ret_row *S* ret.Dim().stride, ret.Dim().stride, S);

   //KALDI_LOG << "sum = " << ret.Sum();

   cuda_make_tran( S/BLOCKSIZE+1, BLOCKSIZE, dim.rows, dim.cols, lab.Data(), l,
         ret.Data() + ret_row *S * ret.Dim().stride, ret.Dim().stride, S);

   //KALDI_LOG << "sum = " << ret.Sum();

   CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());

   //KALDI_LOG << "sum = " << ret.Sum();

   // check generated features.
   //Matrix<BaseFloat> feats_host(feats.NumRows(), feats.NumCols());
   //feats_host.CopyFromMat(feats);

   //KALDI_LOG << "feats sum = " << feats_host.Sum();

   //Matrix<BaseFloat> ret_host(ret.NumRows(), ret.NumCols());
   //ret_host.CopyFromMat(ret);
   //
   //KALDI_LOG << "sum = " << ret_host.Sum();


   //vector<int> lab_host;
   //lab.CopyToVec(lab_host);
   //for(int i = 0; i < lab_host.size(); ++i)
   //   lab_host[i] += 1;

   //Matrix<BaseFloat> cpu(S, F*S + S*S);
   //for(int s = 0; s < S; ++s){
   //   lab_host[l] = s + 1;
   //   makeFeature(feats_host, lab_host, S, cpu.Row(s));
   //}

   //KALDI_LOG << "sum = " << cpu.Sum();


   //float err = 0;
   //for(int i = 0; i < S; ++i)
   //   for(int j = 0; j < cpu.NumCols(); ++j)
   //      err += (cpu(i, j) - ret_host(i+ret_row*S, j))*(cpu(i, j) - ret_host(i+ret_row*S, j));

   //assert(err < 0.01);
}

void makeFeatureCuda(const myCuMatrix<BaseFloat> &feats, const CuIntVector &lab, int S, myCuMatrix<BaseFloat> &ret){
   Timer tim;

   MatrixDim dim = feats.Dim();
   int L = dim.rows;
   int F = dim.cols;

   assert( L == lab.Dim() );

   ret.Resize(L*S, F*S + S*S);

   //KALDI_LOG << "sum = " << ret.Sum();

   //assert( cudaSuccess == cudaGetLastError() );
   cuda_make_obs( (L * S * F)/BLOCKSIZE+1, BLOCKSIZE, feats.Data(), dim.rows, dim.cols,
         dim.stride, lab.Data(), ret.Data(), ret.Dim().stride, S);
   //assert( cudaSuccess == cudaGetLastError() );

   //KALDI_LOG << "sum = " << ret.Sum();

   cuda_make_tran( (L * S)/BLOCKSIZE+1, BLOCKSIZE, dim.rows, dim.cols, lab.Data(), 
         ret.Data(), ret.Dim().stride, S);
   assert( cudaSuccess == cudaGetLastError() );

   // TODO
   //ret.Scale(1/(double)L);

   //KALDI_LOG << "sum = " << ret.Sum();

   // check generated features.
   //Matrix<BaseFloat> feats_host(feats.NumRows(), feats.NumCols());
   //feats_host.CopyFromMat(feats);

   //KALDI_LOG << "feats sum = " << feats_host.Sum();

   //Matrix<BaseFloat> ret_host(ret.NumRows(), ret.NumCols());
   //ret_host.CopyFromMat(ret);
   //
   //KALDI_LOG << "sum = " << ret_host.Sum();


   //vector<int> lab_host;
   //lab.CopyToVec(lab_host);
   //for(int i = 0; i < lab_host.size(); ++i)
   //   lab_host[i] += 1;

   //Matrix<BaseFloat> cpu(L*S, F*S + S*S);
   //for(int l = 0; l < L; ++l){
   //   int old = lab_host[l];
   //   for(int s = 0; s < S; ++s){
   //      lab_host[l] = s + 1;
   //      makeFeature(feats_host, lab_host, S, cpu.Row(l*S + s));
   //   }
   //   lab_host[l] = old;
   //}

   //KALDI_LOG << "sum = " << cpu.Sum();


   //float err = 0;
   //for(int i = 0; i < cpu.NumRows(); ++i)
   //   for(int j = 0; j < cpu.NumCols(); ++j)
   //      err += (cpu(i, j) - ret_host(i, j))*(cpu(i, j) - ret_host(i, j));

   //assert(err < 0.01);

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
