#include "strt.h"

StrtBase::StrtBase(bool pair, double error): pair_(pair), error_(error),
         frames_progress_(0), correct_progress_(0), loss_progress_(0),
         frames_N_(0), diff_host_(2){

   if(pair){
      frames_arr_.resize(END_TYPE * END_TYPE);
      correct_arr_.resize(END_TYPE * END_TYPE);
      loss_arr_.resize(END_TYPE * END_TYPE);
   }else{
      frames_arr_.resize(END_TYPE);
      correct_arr_.resize(END_TYPE);
      loss_arr_.resize(END_TYPE);
   }
}

void StrtBase::Eval(const VectorBase<BaseFloat> &delta, const CuMatrixBase<BaseFloat> &nnet_out,
      vector<CuMatrix<BaseFloat> > *diff_dev, const vector<int>* example_type){

   int N = delta.Dim();

   KALDI_ASSERT(nnet_out.NumCols() == 1);
   KALDI_ASSERT(N ==  nnet_out.NumRows());

   // copy data from gpu to cpu
   nnet_out_host_.Resize(nnet_out.NumRows(), nnet_out.NumCols(), kUndefined);
   nnet_out_host_.CopyFromMat(nnet_out);

   diff_host_[0].Resize(N, 1, kSetZero);
   diff_host_[1].Resize(N, 1, kSetZero);

   double total_loss = 0, total_correct = 0;

   for(int i = 0; i < N; ++i){
      assert(delta(i) != 0);

      BaseFloat f_out = nnet_out_host_(i, 0);
      BaseFloat a_tgt = delta(i);

      BaseFloat loss, diff;
      
      calcLoss(f_out, a_tgt, loss, diff);

      bool correct = pair_ ? f_out * a_tgt > 0 : abs(f_out - a_tgt) < error_;

      diff_host_[0](i, 0) += diff;
      diff_host_[1](i, 0) -= diff;

      if( correct ) total_correct += 1;

      total_loss += loss;

      if(example_type != NULL){
         if( correct ) correct_arr_[(*example_type)[i]] += 1;

         loss_arr_[(*example_type)[i]] += loss;
         frames_arr_[(*example_type)[i]] += 1;
      }
   }

   KALDI_ASSERT(KALDI_ISFINITE(total_loss));

   if(diff_dev != NULL){
      assert(diff_dev -> size() == 2);

      (*diff_dev)[0] = diff_host_[0];
      (*diff_dev)[1] = diff_host_[1];
   }

   frames_arr_[ALL_TYPE]  += N;
   loss_arr_[ALL_TYPE]    += total_loss;
   correct_arr_[ALL_TYPE] += total_correct;

   // progress losss reporting
   {
      static const int32 progress_step = 3600; 
      frames_progress_  += N;
      loss_progress_    += total_loss; 
      correct_progress_ += total_correct;

      if (frames_progress_ > progress_step) {
         KALDI_VLOG(1) << "ProgressLoss[ " 
            << static_cast<int>(frames_arr_[ALL_TYPE]/progress_step) << "h of " 
            << static_cast<int>(frames_N_/progress_step) << "h]: " 
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

}

StrtBase* StrtBase::getInstance(string name, bool pair, double error){
   if(name == "mse"){
      return new StrtMse(pair, error);
   }else if(name == "mgn"){
      return new StrtMgn(pair, error);
   }else if(name == "softmax"){
      return new StrtSoftmax(pair, error);
   }else if(name == "wsoftmax"){
      return new StrtWSoftmax(pair, error);
   }else if(name == "exp"){
      return new StrtExp(pair, error);
   }else{
      return NULL;
   }
}

string StrtBase::Report() {
   ostringstream oss;
   oss << "AvgLoss: " << loss_arr_[ALL_TYPE]/frames_arr_[ALL_TYPE] << " (Strt) " << endl;
   for(int i = 0; i < frames_arr_.size(); ++i){
      if(frames_arr_[i] > 0 && i != ALL_TYPE){
         oss << "  " << getStr(i)
            << " Loss: " << loss_arr_[i] / frames_arr_[i] << " (Strt) " << endl;
      }
   }
      
   if (loss_vec_.size() > 0) {
      oss << "progress: [";
      copy(loss_vec_.begin(),loss_vec_.end(),ostream_iterator<float>(oss," "));
      oss << "]" << endl;
   }

   if (correct_arr_[ALL_TYPE] >= 0) {
      oss << "FRAME_ACCURACY >> " << 100.0*correct_arr_[ALL_TYPE]/frames_arr_[ALL_TYPE] << "% <<" << endl;
      for(int i = 0; i < frames_arr_.size(); ++i){
         if(frames_arr_[i] > 0 && i != ALL_TYPE){
            oss << "  " << getStr(i) 
               << " ACC >> " << 100.0*correct_arr_[i] / frames_arr_[i] << "% << " << endl;
         }
      }
   }
   return oss.str(); 
}

string StrtBase::getStr(int index){
   if(pair_){
      int m = index / END_TYPE;
      int n = index % END_TYPE;
      return LABEL_NAME[m] + " <-> " + LABEL_NAME[n];
   }else{
      return LABEL_NAME[index];
   }
}

typedef struct{
   BaseFloat v1;
   BaseFloat v2;
   int       id;
} Tuple;

bool tuple_cmp(const Tuple &a, const Tuple &b){
   if(a.v1 != b.v1) return a.v1 > b.v1;
   return a.v2 > b.v2;

}

void StrtListBase::Eval(const vector<BaseFloat> &nnet_target,
      const CuMatrixBase<BaseFloat> &nnet_out, CuMatrix<BaseFloat> *diff){

   int N = nnet_target.size();

   KALDI_ASSERT(nnet_out.NumCols() == 1);
   KALDI_ASSERT(N ==  nnet_out.NumRows());

   // copy data from gpu to cpu
   nnet_out_host_.Resize(nnet_out.NumRows(), nnet_out.NumCols(), kUndefined);
   nnet_out_host_.CopyFromMat(nnet_out);

   // sorting
   vector< Tuple > arr(N);
   for(int i = 0; i < N; ++i){
      arr[i].v1 = nnet_target[i];
      arr[i].v2 = nnet_out_host_(i, 0);
      arr[i].id = i;
   }

   sort(arr.begin(), arr.end(), tuple_cmp);

   vector<int> index(N);
   for(int i = 0; i < N; ++i)
      index[i] = arr[i].id;

   // start accumulate.

   double sum = 0;
   for(int i = 0; i < N; ++i)
      sum += nnet_out_host_(i, 0);

   vector<double> acc_sum(N);
   acc_sum[N-1] = nnet_out_host_(index[N-1], 0);
   for(int i = N-2; i >= 0; --i){
      acc_sum[i] = log_add(acc_sum[i+1], nnet_out_host_(index[i], 0));
   }

   // calculating loss
   double loss = -sum;
   for(int i = 0; i < N; ++i){
      loss += acc_sum[i];
   }

   // calculate gradient
   vector<double> acc_sum_log(N);
   acc_sum_log[0] = -acc_sum[0];
   for(int i = 1; i < N; ++i)
      acc_sum_log[i] = log_add(acc_sum_log[i-1], -acc_sum[i]);

   diff_host_.Resize(N, 1);
   for(int i = 0; i < N; ++i){
      diff_host_(index[i], 0) = exp(nnet_out_host_(index[i], 0) + acc_sum_log[i]) - 1;
   }

   sum = 0;
   for(int i = 0; i < N; ++i)
      sum += diff_host_(i, 0);
   
   KALDI_ASSERT(KALDI_ISFINITE(sum));
   KALDI_ASSERT(KALDI_ISFINITE(loss));

   if(diff != NULL){
      *diff = diff_host_;
   }

   int correct = 1;
   for(int i = 0; i < N; ++i){
      if(index[i] != i){
         correct = 0;
         break;
      }
   }

   frames_  += 1;
   loss_    += loss;
   correct_ += correct;

   // progress losss reporting
   {
      static const int32 progress_step = 10; 
      frames_progress_  += 1;
      loss_progress_    += loss; 
      correct_progress_ += correct;

      if (frames_progress_ > progress_step) {
         KALDI_VLOG(1) << "ProgressLoss[ " 
            << static_cast<int>(frames_/progress_step) << "d of " 
            << static_cast<int>(frames_N_/progress_step) << "d]: " 
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
}

string StrtListBase::Report(){
   ostringstream oss;
   oss << "AvgLoss: " << loss_/frames_ << " (Strt) " << endl;
      
   if (loss_vec_.size() > 0) {
      oss << "progress: [";
      copy(loss_vec_.begin(),loss_vec_.end(),ostream_iterator<float>(oss," "));
      oss << "]" << endl;
   }

   if (correct_ >= 0) {
      oss << "FRAME_ACCURACY >> " << 100.0*correct_/frames_ << "% <<" << endl;
   }
   return oss.str(); 
}

void StrtMse::calcLoss(BaseFloat f_out, BaseFloat a_tgt,
      BaseFloat &loss, BaseFloat &diff){

   double q = f_out - a_tgt;
   loss = q * q / 2;
   diff = q;
}

void StrtMgn::calcLoss(BaseFloat f_out, BaseFloat a_tgt,
      BaseFloat &loss, BaseFloat &diff){

   int sign_a = a_tgt > 0 ? 1 : -1;
   double q = sign_a * ( a_tgt - f_out );

   if(q > 0){
      loss = q;
      diff = -sign_a;

   }else{
      loss = 0;
      diff = 0;
   }
}

void StrtSoftmax::calcLoss(BaseFloat f_out, BaseFloat a_tgt,
      BaseFloat &loss, BaseFloat &diff){

   double tgt1 = a_tgt > 0 ? 1: 0;
   double tgt2 = a_tgt > 0 ? 0: 1;

   double soft1 = sigmoid(f_out); 
   double soft2 = 1 - soft1;

   // avoid log(0)
   soft1 += FLT_MIN; 
   soft2 += FLT_MIN; 

   loss = - tgt1 * log(soft1) - tgt2 * log(soft2);
   diff = - tgt1 * soft2 + tgt2 * soft1;
}

void StrtWSoftmax::calcLoss(BaseFloat f_out, BaseFloat a_tgt,
      BaseFloat &loss, BaseFloat &diff){

   double tgt1 = a_tgt > 0 ? 1: 0;
   double tgt2 = a_tgt > 0 ? 0: 1;

   double soft1 = sigmoid(f_out);
   double soft2 = 1 - soft1;

   // avoid log(0)
   soft1 += FLT_MIN; 
   soft2 += FLT_MIN; 

   loss = (- tgt1 * log(soft1) - tgt2 * log(soft2) ) *abs(a_tgt);
   diff = (- tgt1 * soft2 + tgt2 * soft1) *abs(a_tgt);
}

void StrtExp::calcLoss(BaseFloat f_out, BaseFloat a_tgt,
      BaseFloat &loss, BaseFloat &diff){

   int sign_a = a_tgt > 0 ? 1 : -1;
   loss = exp(-sign_a * f_out);
   diff = -sign_a * loss;
}
