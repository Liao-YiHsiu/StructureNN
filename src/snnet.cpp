#include "snnet.h"

SNnet & SNnet::operator = (const SNnet &other){
   nnet1_ = other.nnet1_;
   nnet2_ = other.nnet2_;
   stateMax_ = other.stateMax_;
   return *this;
}

SNnet::~SNnet(){
   Destroy();
}


void SNnet::Propagate(const vector<CuMatrix<BaseFloat>* > &in_arr, 
      const vector<vector<uchar>* > &labels, CuMatrix<BaseFloat> *out){

   // feedforward the first pass
   if(transf_arr_.size() != in_arr.size()) transf_arr_.resize(in_arr.size());
   if(propagate_buf_.size() != in_arr.size()) propagate_buf_.resize(in_arr.size());

   for(int i = 0; i < in_arr.size(); ++i)
      nnet_transf_.Feedforward(*in_arr[i], &transf_arr_[i]);

   nnet1_.Propagate(transf_arr_, propagate_buf_);

   // combine each buffer according to label
   Psi(propagate_buf_, labels, &psi_buff_);

   nnet2_.Propagate(psi_buff_, out);

   if (!KALDI_ISFINITE(out->Sum())) { // check there's no nan/inf,
      KALDI_ERR << "NaN or inf found in nn-output nnet2";
   }
}

void SNnet::Backpropagate(const CuMatrix<BaseFloat> &out_diff,
      const vector<vector<uchar>* > &labels){

   if(backpropagate_buf_.size() != labels.size()) backpropagate_buf_.resize(labels.size());

   nnet2_.Backpropagate(out_diff, &psi_diff_);

   BackPsi(psi_diff_, labels, backpropagate_buf_);

   if(propagate_buf_.size() == 1){
      for(int i = 1; i < backpropagate_buf_.size(); ++i)
         backpropagate_buf_[0].AddMat(1.0, backpropagate_buf_[i]);
      nnet1_.Backpropagate(backpropagate_buf_[0], NULL);
   }else{
      nnet1_.Backpropagate(backpropagate_buf_);
   }

}

void SNnet::Propagate(const vector<CuMatrix<BaseFloat>* > &in_arr,
      const vector<vector<uchar>* > &labels,
      const vector<vector<uchar>* > &ref_labels, CuMatrix<BaseFloat> *out) {

   int N = in_arr.size();

   // feedforward the first pass
   if(transf_arr_.size() != N) transf_arr_.resize(N);
   if(propagate_buf_.size() != N) propagate_buf_.resize(N);

   for(int i = 0; i < N; ++i)
      nnet_transf_.Feedforward(*in_arr[i], &transf_arr_[i]);

   nnet1_.Propagate(transf_arr_, propagate_buf_);

   if(psi_arr_.size() != 2) psi_arr_.resize(2);
   if(out_arr_.size() != 2) out_arr_.resize(2);

   // combine each buffer according to label
   Psi(propagate_buf_, labels, &psi_arr_[0]);
   Psi(propagate_buf_, ref_labels, &psi_arr_[1]);

   nnet2_.Propagate(psi_arr_, out_arr_);
   
   *out = out_arr_[0];
   out -> AddMat(-1, out_arr_[1]);

   if (!KALDI_ISFINITE(out->Sum())) { // check there's no nan/inf,
      KALDI_ERR << "NaN or inf found in nn-output nnet2";
   }
}

void SNnet::Backpropagate(const vector<CuMatrix<BaseFloat> > &out_diff, 
      const vector<vector<uchar>* > &labels,
      const vector<vector<uchar>* > &ref_labels){

   KALDI_ASSERT( out_diff.size() == 2);

   if(psi_diff_arr_.size() != 2) psi_diff_arr_.resize(2);

   nnet2_.Backpropagate(out_diff, &psi_diff_arr_);

   if(backpropagate_arr_.size() != 2) backpropagate_arr_.resize(2);

   if(backpropagate_arr_[0].size() != labels.size()) backpropagate_arr_[0].resize(labels.size());
   if(backpropagate_arr_[1].size() != labels.size()) backpropagate_arr_[1].resize(labels.size());


   BackPsi(psi_diff_arr_[0], labels,     backpropagate_arr_[0]);
   BackPsi(psi_diff_arr_[1], ref_labels, backpropagate_arr_[1]);

   // merge backpropagate_arr[0, 1] into one matrix
   for(int i = 0; i < backpropagate_arr_[0].size(); ++i){
      backpropagate_arr_[0][i].AddMat(1, backpropagate_arr_[1][i]);
   }

   nnet1_.Backpropagate(backpropagate_arr_[0]);
}

void SNnet::Feedforward(const vector<CuMatrix<BaseFloat>* > &in_arr,
      const vector<vector<uchar>* > &labels, CuMatrix<BaseFloat> *out){
   // feedforward the first pass
   if(transf_arr_.size() != in_arr.size()) transf_arr_.resize(in_arr.size());
   if(propagate_buf_.size() != in_arr.size()) propagate_buf_.resize(in_arr.size());

   for(int i = 0; i < in_arr.size(); ++i)
      nnet_transf_.Feedforward(*in_arr[i], &transf_arr_[i]);

   nnet1_.Feedforward(transf_arr_, propagate_buf_);

   // combine each buffer according to label
   Psi(propagate_buf_, labels, &psi_buff_);

   nnet2_.Feedforward(psi_buff_, out);

   if (!KALDI_ISFINITE(out->Sum())) { // check there's no nan/inf,
      KALDI_ERR << "NaN or inf found in nn-output nnet2";
   }
}

void SNnet::Feedforward(const CuMatrix<BaseFloat> &in, const vector<vector<uchar>* > &labels, CuMatrix<BaseFloat> *out){
   // feedforward the first pass
   if(transf_arr_.size() != 1)     transf_arr_.resize(1);
   if(propagate_buf_.size() != 1)  propagate_buf_.resize(1);

   nnet_transf_.Feedforward(in, &transf_arr_[0]);

   nnet1_.Feedforward(transf_arr_[0], &propagate_buf_[0]);

   // combine each buffer according to label
   Psi(propagate_buf_, labels, &psi_buff_);

   nnet2_.Feedforward(psi_buff_, out);

   if (!KALDI_ISFINITE(out->Sum())) { // check there's no nan/inf,
      KALDI_ERR << "NaN or inf found in nn-output nnet2";
   }
}

void SNnet::Propagate(const CuMatrix<BaseFloat> &in, const vector<vector<uchar>* > &labels, CuMatrix<BaseFloat> *out){
   // feedforward the first pass
   if(transf_arr_.size() != 1)     transf_arr_.resize(1);
   if(propagate_buf_.size() != 1)  propagate_buf_.resize(1);

   nnet_transf_.Feedforward(in, &transf_arr_[0]);

   nnet1_.Propagate(transf_arr_[0], &propagate_buf_[0]);

   // combine each buffer according to label
   Psi(propagate_buf_, labels, &psi_buff_);

   nnet2_.Propagate(psi_buff_, out);

   if (!KALDI_ISFINITE(out->Sum())) { // check there's no nan/inf,
      KALDI_ERR << "NaN or inf found in nn-output nnet2";
   }
}

void SNnet::Acc(const CuMatrix<BaseFloat> &in, const vector<vector<uchar>* > &labels){
   // feedforward the first pass
   if(transf_arr_.size() != 1)     transf_arr_.resize(1);
   if(propagate_buf_.size() != 1)  propagate_buf_.resize(1);

   nnet_transf_.Feedforward(in, &transf_arr_[0]);

   nnet1_.Propagate(transf_arr_[0], &propagate_buf_[0]);

   // combine each buffer according to label
   Psi(propagate_buf_, labels, &psi_buff_);

   // compute mean and variance from psi_buff_
   if(stat_aux_.NumRows() != psi_buff_.NumCols() || stat_aux_.NumCols() != 1 ){
      stat_aux_.Resize(1, psi_buff_.NumRows(), kSetZero);
      stat_aux_.Set(1);
   }

   if(stat_sum_.NumCols() != psi_buff_.NumCols() || stat_sum_.NumRows() != 1 ||
         stat_sqr_.NumCols() != psi_buff_.NumCols() || stat_sqr_.NumRows() != 1){
      stat_sum_.Resize(1, psi_buff_.NumCols(), kSetZero);
      stat_sqr_.Resize(1, psi_buff_.NumCols(), kSetZero);
      stat_N_ = 0;
   }

   stat_sum_.AddMatMat(1, stat_aux_, kNoTrans, psi_buff_, kNoTrans, 1);

   psi_buff_.ApplyPow(2);
   stat_sqr_.AddMatMat(1, stat_aux_, kNoTrans, psi_buff_, kNoTrans, 1);

   stat_N_ += psi_buff_.NumRows();
}

void SNnet::Stat(CuVector<BaseFloat> &mean, CuVector<BaseFloat> &sd){
   if(mean.Dim() != stat_sum_.NumCols()) mean.Resize(stat_sum_.NumCols(), kUndefined);
   mean.CopyRowsFromMat(stat_sum_);

   mean.Scale(1.0/stat_N_);

   CuVector<BaseFloat> mean_sqr(mean);
   mean_sqr.ApplyPow(2);

   if(sd.Dim() != stat_sum_.NumCols()) sd.Resize(stat_sum_.NumCols(), kUndefined);
   sd.CopyRowsFromMat(stat_sqr_);

   sd.Scale(1.0/stat_N_);
   sd.AddVec(-1, mean_sqr);

   sd.ApplyPow(0.5);
}

void SNnet::PropagatePsi(const CuMatrix<BaseFloat> &in,
      const vector<vector<uchar>* > &labels, CuMatrix<BaseFloat> *out){

   // feedforward the first pass
   if(transf_arr_.size() != 1)     transf_arr_.resize(1);
   if(propagate_buf_.size() != 1)  propagate_buf_.resize(1);

   nnet_transf_.Feedforward(in, &transf_arr_[0]);

   nnet1_.Propagate(transf_arr_[0], &propagate_buf_[0]);

   // combine each buffer according to label
   Psi(propagate_buf_, labels, out);

   if (!KALDI_ISFINITE(out->Sum())) { // check there's no nan/inf,
      KALDI_ERR << "NaN or inf found in nn-output nnet2";
   }
}


int32 SNnet::InputDim() const{
   return nnet1_.InputDim();
}

int32 SNnet::OutputDim() const{
   return nnet2_.OutputDim();
}

int32 SNnet::NumParams() const{
   return nnet1_.NumParams() + nnet2_.NumParams();
}

void SNnet::SetDropoutRetention(BaseFloat r){
   nnet1_.SetDropoutRetention(r);
   nnet2_.SetDropoutRetention(r);
}

void SNnet::Init(const string &config_file1, const string &config_file2, uchar stateMax) {
   nnet1_.Init(config_file1);
   nnet2_.Init(config_file2);
   stateMax_ = stateMax;

   Check();
}

void SNnet::Read(const string &file1, const string &file2, uchar stateMax) {
   nnet1_.Read(file1);
   nnet2_.Read(file2);
   stateMax_ = stateMax;
   Check();
} 

void SNnet::Write(const string &file1, const string &file2, bool binary) const {
   Check();
   nnet1_.Write(file1, binary);
   nnet2_.Write(file2, binary);
}

void SNnet::Read(const string &file1, uchar stateMax) {
   nnet1_.Read(file1);
   stateMax_ = stateMax;
} 

string SNnet::Info() const {
   return "Nnet1\n" + nnet1_.Info() + "Nnet2\n" + nnet2_.Info();
}

string SNnet::InfoGradient() const {
   return "Nnet1\n" + nnet1_.InfoGradient() + "Nnet2\n" + nnet2_.InfoGradient();
}

string SNnet::InfoPropagate() const {
   return "Nnet1\n" + nnet1_.InfoPropagate() + "Nnet2\n" + nnet2_.InfoPropagate();
}
string SNnet::InfoBackPropagate() const {
   return "Nnet1\n" + nnet1_.InfoBackPropagate() + "Nnet2\n" + nnet2_.InfoBackPropagate();
}

void SNnet::Check() const {
   KALDI_ASSERT(((nnet1_.OutputDim() + 1) * (stateMax_ + 1) + stateMax_ * stateMax_) == nnet2_.InputDim());
}

void SNnet::Destroy() {

   psi_buff_.Resize(0, 0);
   psi_diff_.Resize(0, 0);
}

void SNnet::SetTrainOptions(const NnetTrainOptions& opts, double ratio) {
   NnetTrainOptions opts_tmp = opts;
   opts_tmp.learn_rate *= ratio;
   nnet1_.SetTrainOptions(opts_tmp);
   nnet2_.SetTrainOptions(opts);
}

const NnetTrainOptions& SNnet::GetTrainOptions() const {
   return nnet2_.GetTrainOptions();
}

void SNnet::SetTransform(const Nnet &nnet){
   nnet_transf_ = nnet;
}


void SNnet::Psi(vector<CuMatrix<BaseFloat> > &feats, const vector<vector<uchar>* > &labels,
      CuMatrix<BaseFloat> *out){

   PsiKernel(propagate_buf_, labels, out);

   //KALDI_ASSERT(out != NULL);
   //KALDI_ASSERT(feats.size() == labels.size() || feats.size() == 1);

   //int N = labels.size();
   //int F = feats[0].NumCols();

   //out->Resize(N, (F + 1)*(stateMax_ + 1) + stateMax_ * stateMax_, kSetZero);

   //for(int i = 0; i < N; ++i){
   //   if(feats.size() == 1)
   //      makeFeat(feats[0], *labels[i], out->Row(i));
   //   else
   //      makeFeat(feats[i], *labels[i], out->Row(i));
   //}

   //// check
   //CuMatrix<BaseFloat> tmp;
   //PsiKernel(propagate_buf_, labels, &tmp);
   //tmp.AddMat(-1, *out);
   //tmp.MulElements(tmp);
   //assert(tmp.Sum() < 1e-10);
}

void SNnet::PsiKernel(vector<CuMatrix<BaseFloat> > &feats, const vector<vector<uchar>* > &labels,
      CuMatrix<BaseFloat> *out){

   KALDI_ASSERT(out != NULL);
   KALDI_ASSERT(feats.size() == labels.size() || feats.size() == 1);
   KALDI_ASSERT(feats[0].NumCols() == nnet1_.OutputDim());

   int N = labels.size();
   int F = feats[0].NumCols();

   int obs_len   = F + stateMax_ * F;
   int trans_len = 1 + stateMax_ + stateMax_ * stateMax_;

   out->Resize(N, obs_len + trans_len, kSetZero);

   // prepare packs_device
   int maxL = packPsi(feats, labels, *out, packs_device_);

   // use kernel to propagate psi
   propPsi(N, F, stateMax_, maxL, packs_device_.Data());

   // build transition using cpu
   Matrix<BaseFloat> trans(N, trans_len, kSetZero);
#pragma omp parallel for
   for(int i = 0; i < N; ++i){
      const vector<uchar>& lab = *labels[i];
      for(int j = 0; j < lab.size(); ++j){
         trans(i, 0) += 1;
         trans(i, lab[j]) += 1;
         if(j != 0) trans(i, lab[j-1] * stateMax_ + lab[j] ) += 1;
      }
   }

   CuSubMatrix<BaseFloat> sub = out->Range(0, N, obs_len, trans_len);
   sub.CopyFromMat(trans);
}

int SNnet::packPsi(vector<CuMatrix<BaseFloat> > &feats, const vector<vector<uchar>* > &labels, CuMatrix<BaseFloat> &psi_feats, CuVectorG<PsiPack> &packs_dev){
   KALDI_ASSERT(feats.size() == labels.size() || feats.size() == 1);
   KALDI_ASSERT(feats[0].NumCols() == nnet1_.OutputDim());

   int N = labels.size();

   // find max L from previous L
   int maxL = labelbuf_cols_;
   for(int i = 0; i < labels.size(); ++i)
      maxL = labels[i]->size() > maxL ? labels[i]->size() : maxL;
   labelbuf_cols_ = maxL;
   
   if(labelbuf_.size() != N * labelbuf_cols_) 
      labelbuf_.resize(N * labelbuf_cols_);

   for(int i = 0; i < N; ++i)
      for(int j = 0; j < labels[i]->size(); ++j)
         labelbuf_[i*labelbuf_cols_ + j] = (*labels[i])[j];

   labelbuf_device_.Resize(N, maxL);
   labelbuf_device_.CopyFromVec(labelbuf_);

   // copy labels into labelbuf.
   if(packs_.size() != N) packs_.resize(N);
   for(int i = 0; i < N; ++i){
      PsiPack &p = packs_[i];
      p.L           = labels[i]->size();
      p.lab         = labelbuf_device_.Data() + i * labelbuf_cols_;
      p.feat        = feats.size() == 1 ? feats[0].Data(): feats[i].Data();
      p.feat_stride = feats.size() == 1 ? feats[0].Stride(): feats[i].Stride();
      p.psi_feat    = psi_feats.Data() + i * psi_feats.Stride();
   }

   packs_dev.CopyFromVec(packs_);

   return maxL;
}


void SNnet::BackPsi(CuMatrix<BaseFloat> &diff, const vector<vector<uchar>* > &labels, vector<CuMatrix<BaseFloat> > &feats_diff){

   BackPsiKernel(diff, labels, feats_diff);

   //KALDI_ASSERT(feats_diff.size() == labels.size());

   //for(int i = 0; i < labels.size(); ++i){
   //   feats_diff[i].Resize(labels[i]->size(), nnet1_.OutputDim(), kSetZero);
   //   distErr(diff.Row(i), *labels[i], feats_diff[i]);
   //}

   //vector< CuMatrix<BaseFloat> > tmp(feats_diff.size());
   //BackPsiKernel(diff, labels, tmp);

   //for(int i = 0; i < feats_diff.size(); ++i){
   //   tmp[i].AddMat(-1, feats_diff[i]);
   //   tmp[i].MulElements(tmp[i]);
   //   assert(tmp[i].Sum() < 1e-10);
   //}
}


// TODO: unnormalized
// cut it into 2 parts. cpu parts(transitions) and gpu parts(observations)
void SNnet::makeFeat(CuMatrix<BaseFloat> &feat, const vector<uchar> &label, CuSubVector<BaseFloat> vec) {
   KALDI_ASSERT(feat.NumRows() == label.size());

   int F = feat.NumCols();

   int obs_len   = F * (stateMax_ + 1);
   int trans_len = 1 + stateMax_ + stateMax_ * stateMax_;

   // construct observation

   for(int i = 0; i < label.size(); ++i){
      KALDI_ASSERT(label[i] <= stateMax_);
      if(label[i] == 0) continue;

      CuSubVector<BaseFloat> obs(vec, label[i]*F, F);
      CuSubVector<BaseFloat> dummy(vec, 0, F);

      obs.AddVec(1, feat.Row(i));
      dummy.AddVec(1, feat.Row(i));
   }

   // construct transition
   Vector<BaseFloat> trans(trans_len, kSetZero);
   for(int i = 0; i < label.size(); ++i){
      trans(0) += 1;
      trans(label[i]) += 1;
      if(i != 0) trans(label[i-1] * stateMax_ + label[i]) += 1;
   }

   CuSubVector<BaseFloat> trans_gpu(vec, obs_len, trans_len);
   trans_gpu.CopyFromVec(trans);
}

// distribute Error into some matrix
void SNnet::distErr(const CuSubVector<BaseFloat> &diff, const vector<uchar>& label, CuMatrix<BaseFloat> &mat){
   KALDI_ASSERT(mat.NumRows() == label.size());
   KALDI_ASSERT(mat.NumCols() == nnet1_.OutputDim());

   int F = mat.NumCols();

   CuSubVector<BaseFloat> dummy(diff, 0, F);

   for(int i = 0; i < label.size(); ++i){
      KALDI_ASSERT(label[i] <= stateMax_);
      if(label[i] == 0) continue;

      CuSubVector<BaseFloat> obs(diff, label[i]*F, F);

      mat.Row(i).AddVec(1, dummy);
      mat.Row(i).AddVec(1, obs);
   }
}

void SNnet::BackPsiKernel(CuMatrix<BaseFloat> &diff, const vector<vector<uchar>* > &labels, vector<CuMatrix<BaseFloat> > &feats_diff){

   KALDI_ASSERT(feats_diff.size() == labels.size());

   // reserve space for backpropagation
   for(int i = 0; i < labels.size(); ++i){
      feats_diff[i].Resize(labels[i]->size(), nnet1_.OutputDim(), kSetZero);
   }

   int N = labels.size();
   int F = nnet1_.OutputDim();

   // prepare packs_device
   int maxL = packPsi(feats_diff, labels, diff, packs_device_);

   // use kernel to build psi
   backPsi(N, F, stateMax_, maxL, packs_device_.Data());
}
