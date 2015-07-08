#include "snnet.h"

SNnet & SNnet::operator = (const SNnet &other){
   nnet1_ = other.nnet1_;
   nnet2_ = other.nnet2_;
   stateMax_ = other.stateMax_;
}

SNnet::~SNnet(){
   Destroy();
}


void SNnet::Propagate(const vector<CuMatrix<BaseFloat>* > &in_arr, 
      const vector<vector<int32> > &labels, CuMatrix<BaseFloat> *out){

   assert(in_arr.size() == labels.size());
   if( propagate_buf_.size() != in_arr.size() ){
      propagate_buf_.resize(in_arr.size());
      backpropagate_buf_.resize(in_arr.size());
   }

   for(int i = 0; i < in_arr.size(); ++i){
      nnet1_.Propagate(*in_arr[i], &propagate_buf_[i], i);
   }

   // combine each buffer according to label
   Psi(propagate_buf_, labels, &psi_buff_);

   nnet2_.Propagate(psi_buff_, out);

   labels_ = labels;
}

void SNnet::Backpropagate(const CuMatrix<BaseFloat> &out_diff){

   nnet2_.Backpropagate(out_diff, &psi_diff_);

   BackPsi(psi_diff_, labels_, &backpropaget_buf_);

   for(int i = 0; i < backpropagate_buf_.size(); ++i)
      nnet1_.Backpropagate(backpropagate_buf_[i], NULL, i);

   nnet1_.Update();
}

void SNnet::Feedforward(const vector<CuMatrix<BaseFloat>* > &in_arr, const vector<vector<int32> > &labels, CuMatrix<BaseFloat> *out){

   assert(in_arr.size() == labels.size());

   if( propagate_buf_.size() != in_arr.size() ){
      propagate_buf_.resize(in_arr.size());
      backpropagate_buf_.resize(in_arr.size());
   }

   for(int i = 0; i < in_arr.size(); ++i){
      nnet1_.Feedforward(*in_arr[i], &propagate_buf_[i]);
   }

   // combine each buffer according to label
   Psi(propagate_buf_, labels, &psi_buff_);

   nnet2_.Feedforward(psi_buff_, out);
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

void SNnet::Init(const string &config_file1, const string &config_file2, int stateMax) {
   nnet1_.Init(config_file1);
   nnet2_.Init(config_file2);
   stateMax_ = stateMax;

   Check();
}

void SNnet::Read(const string &file1, const string &file2, int stateMax) {
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

string SNnet::Info() const {
   return "Nnet1\n" + nnet1_.Info() + "Nnet2\n" + nnet2_.Info();
}

string SNnet::InfoGradient() const {
   return "Nnet1\n" + nnet1_.InfoGradient() + "Nnet2\n" + nnet2_.InfoGradient();
}

void SNnet::Check() const {
   KALDI_ASSERT(propagate_buf_.size() == backpropagate_buf_.size());
   KALDI_ASSERT(labels_.size() == propagate_buf_.size());

   KALDI_ASSERT(((nnet1_.OutputDim() + 1) * (stateMax_ + 1) + stateMax_ * stateMax_) == nnet2_.InputDim());
}

void SNnet::Destroy() {
   propagate_buf_.resize(0);
   backpropagate_buf_.resize(0);

   labels_.resize(0);

   psi_buff_.Resize(0, 0);
   psi_diff_.Resize(0, 0);
}

void SNnet::SetTrainOptions(const NnetTrainOptions& opts) {
   nnet1_.SetTrainOptions(opts);
   nnet2_.SetTrainOptions(opts);
}

const SNnet::NnetTrainOptions& GetTrainOptions() const {
   return nnet2_.GetTrainOptions();
}


void SNnet::Psi(vector<CuMatrix<BaseFloat> > &feats, const vector<vector<int32> > &labels, CuMatrix<BaseFloat> *out){
   KALDI_ASSERT(out != NULL);
   KALDI_ASSERT(feats.size() == labels.size());

   int N = feats.size();
   int F = feats[0].NumCols();

   out->Resize(N, (F + 1)*(stateMax_ + 1) + stateMax_ * stateMax_, kSetZero);

   for(int i = 0; i < N; ++i){
      makeFeat(feats[i], labels[i], out->Row(i));
   }
}

void SNnet::BackPsi(const CuMatrix<BaseFloat> &diff, const vector<vector<int32> > &labels, vector<CuMatrix<BaseFloat> > &feats_diff){

   KALDI_ASSERT(feats_diff != NULL);
   KALDI_ASSERT(feats_diff->size() == labels.size());

   for(int i = 0; i < labels.size(); ++i){
      feats_diff[i].Resize(labels[i].size(), nnet1_.OutputDim(), kSetZero);
      distErr(diff.Row(i), labels[i], feats_diff[i]);
   }
}


// TODO: unnormalized
// cut it into 2 parts. cpu parts(transitions) and gpu parts(observations)
void SNnet::makeFeat(CuMatrix<BaseFloat> &feat, const vector<int32> &label, CuSubVector<BaseFloat> &vec) {
   KALDI_ASSERT(feat.NumRows() == label.size());

   int F = feat.NumCols();

   int obs_len   = F * (stateMax_ + 1);
   int trans_len = 1 + stateMax_ + stateMax_ * stateMax_;

   // construct observation

   for(int i = 0; i < label.size(); ++i){
      KALDI_ASSERT(label[i] <= stateMax_);

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

void SNnet::distErr(const CuSubVector<BaseFloat> &diff, const vector<int32>& label, CuMatrix<BaseFloat> &mat){
   KALDI_ASSERT(mat.NumRows() == label.size());
   KALDI_ASSERT(mat.NumCols() == nnet1_.OutputDim());

   int F = mat.NumCols();

   CuSubVector<BaseFloat> dummy(diff, 0, F);

   for(int i = 0; i < label.size(); ++i){
      CuSubVector<BaseFloat> obs(diff, label[i]*F, f);

      mat.Row(i).AddVec(1, dummy);
      mat.Row(i).AddVec(1, obs);
   }
}
