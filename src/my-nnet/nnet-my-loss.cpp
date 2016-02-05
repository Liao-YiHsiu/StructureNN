#include "nnet-my-loss.h"

const struct LabelLossBase::my_key_value LabelLossBase::myMarkerMap[] = {
   {LabelLossBase::lList,  "<LabelListLoss>"},
   {LabelLossBase::lFrame, "<LabelFrameLoss>"},
};

const char* LabelLossBase::myTypeToMarker(LabelLossBase::MyLossType t){
   int32 N = sizeof(myMarkerMap)/sizeof(myMarkerMap[0]);
   for(int i = 0; i < N; ++i)
      if( myMarkerMap[i].key == t)
         return myMarkerMap[i].value;
   assert(false);
   return NULL;
}

LabelLossBase::MyLossType LabelLossBase::myMarkerToType(const string &s){
   string s_lower(s);
   transform(s_lower.begin(), s_lower.end(), s_lower.begin(), ::tolower);
   int32 N = sizeof(myMarkerMap)/sizeof(myMarkerMap[0]);
   for(int i = 0; i < N; ++i){
      string m(myMarkerMap[i].value);
      string m_lower(m);
      transform(m_lower.begin(), m_lower.end(), m_lower.begin(), ::tolower);
      if( m_lower == s_lower )
         return myMarkerMap[i].key;
   }

   return lUnknown;
}

LabelLossBase* LabelLossBase::Read(const string &file){
   Input in(file);
   istream &is = in.Stream();

   vector<LabelLossBase*> loss_arr;
   vector<BaseFloat>      loss_weight;
   string conf_line, token;

   while(!is.eof()){
      assert(is.good());
      getline(is, conf_line);
      trim(conf_line);

      if(conf_line == "") continue;
      KALDI_VLOG(1) << conf_line;
      

      BaseFloat weight = 1.0;
      istringstream tis(conf_line + "\n");
      ReadToken(tis, false, &token);
      if(token == "<weight>"){
         ReadBasicType(tis, false, &weight);
         getline(tis, conf_line);
      }

      loss_arr.push_back(GetInstance(conf_line + "\n"));
      loss_weight.push_back(weight);
   }

   if(loss_arr.size() == 0)
      return NULL;
   else if(loss_arr.size() == 1)
      return loss_arr[0];
   else{
      return new LabelMultiLoss(loss_arr, loss_weight);
   }
}

LabelLossBase* LabelLossBase::GetInstance(const string &conf_line){
   istringstream is(conf_line);
   string loss_type_string;

   ReadToken(is, false, &loss_type_string);
   MyLossType type = myMarkerToType(loss_type_string);

   if(type == lUnknown) return NULL;

   LabelLossBase *ans = NewMyLossOfType(type);
   ans->SetParam(is);

   return ans;
}

LabelLossBase* LabelLossBase::NewMyLossOfType(MyLossType type){
   LabelLossBase *ans = NULL;
   switch(type){
      case lList:
         ans = new LabelListLoss();
         break;
      case lFrame:
         ans = new LabelFrameLoss();
         break;
      default:
         assert(false);
   }

   return ans;
}


// ------------------------------------------------------------------------------------------------

void LabelListLoss::SetParam(istream &is){
   string type, token;

   ReadToken(is, false, &type);
   assert(type == "listnet");

   while(!is.eof()){
      ReadToken(is, false, &token);
           if(token == "<TempT>")  ReadBasicType(is, false, &temp_t_);
      else if(token == "<TempY>")  ReadBasicType(is, false, &temp_y_);
      else KALDI_ERR << "Unknown token " << token << ", a typo in config?"
         << " (Temp)";
      is >> ws;
   }

   assert(temp_t_ > 0);
   assert(temp_y_ > 0);
}

void LabelListLoss::Eval(const vector<uchar> &ref, const vector< vector<uchar> > &labels, const CuMatrixBase<BaseFloat> &nnet_out, MyCuMatrix<BaseFloat> *nnet_out_diff){

   int L = labels.size();
   int T = ref.size();

   assert(nnet_out.NumCols() == 1);
   assert(nnet_out.NumRows() % L == 0);
   assert(nnet_out_diff != NULL);

   Matrix<BaseFloat> nnet_out_host(nnet_out.NumRows(), nnet_out.NumCols(), kUndefined);
   nnet_out_host.CopyFromMat(nnet_out);

   //vector<BaseFloat> targets(L);
   // sum up over all time frame
   Vector<BaseFloat> nnet_out_mixed(L);
   Vector<BaseFloat> targets(L);

#pragma omp parallel for
   for(int i = 0; i < L; ++i){
      targets(i) = phone_acc(ref, labels[i], true);

      for(int j = 0; j < T; ++j)
         nnet_out_mixed(i) += nnet_out_host(i + j*L, 0);

      targets(i) *= temp_t_;
      nnet_out_mixed(i) *= temp_y_;

   }

   double max_t = -FLT_MAX, max_n = -FLT_MAX;
   int    max_t_id = -1, max_n_id = -1;
   for(int i = 0; i < L; ++i){
      if(max_t <= targets(i)){
         max_t_id = i;
         max_t = targets(i);
      }
      if(max_n <= nnet_out_mixed(i)){
         max_n_id = i;
         max_n = nnet_out_mixed(i);
      }
   }
   targets.ApplySoftMax();
   nnet_out_mixed.ApplySoftMax();

   // <differential>
   // Cross entropy
   Vector<BaseFloat> nnet_diff_mixed(L);

   // cross-entropy + softmax
   nnet_diff_mixed = nnet_out_mixed;
   nnet_diff_mixed.AddVec(-1.0, targets);

   Matrix<BaseFloat> nnet_diff_host(nnet_out.NumRows(), nnet_out.NumCols());
   // dist error to all time frame
   for(int j = 0; j < T; ++j)
      for(int i = 0; i < L; ++i)
         nnet_diff_host(i + j*L, 0) = nnet_diff_mixed(i);

   nnet_out_diff->Resize(nnet_out.NumRows(), nnet_out.NumCols());
   nnet_out_diff->CopyFromMat(nnet_diff_host);
   // </differential>

   double correct = (max_n_id == max_t_id)?1:0; 

   Vector<BaseFloat> xentropy_aux;
   // calculate cross_entropy (in GPU),
   xentropy_aux = nnet_out_mixed; // y
   xentropy_aux.Add(1e-20); // avoid log(0)
   xentropy_aux.ApplyLog(); // log(y)
   xentropy_aux.MulElements(targets); // t*log(y)
   double cross_entropy = -xentropy_aux.Sum();

   Vector<BaseFloat> entropy_aux;
   // caluculate entropy (in GPU),
   entropy_aux = targets; // t
   entropy_aux.Add(1e-20); // avoid log(0)
   entropy_aux.ApplyLog(); // log(t)
   entropy_aux.MulElements(targets); // t*log(t)
   double entropy = -entropy_aux.Sum();

   KALDI_ASSERT(KALDI_ISFINITE(cross_entropy));
   KALDI_ASSERT(KALDI_ISFINITE(entropy));

   loss_    += cross_entropy;
   entropy_ += entropy;
   correct_ += correct;
   frames_  += 1;

   // progressive loss reporting
   {
      static const int32 progress_step = 200;
      frames_progress_ += 1;
      loss_progress_ += cross_entropy;
      entropy_progress_ += entropy;
      if (frames_progress_ > progress_step) {
         KALDI_VLOG(1) << "ProgressLoss[last " 
            << static_cast<int>(frames_progress_/200) << "h of " 
            << static_cast<int>(frames_/200) << "h]: " 
            << (loss_progress_-entropy_progress_)/frames_progress_ << " (ListNet)";
         // store
         loss_vec_.push_back((loss_progress_-entropy_progress_)/frames_progress_);
         // reset
         frames_progress_ = 0;
         loss_progress_ = 0.0;
         entropy_progress_ = 0.0;
      }
   }
}

string LabelListLoss::Report(){
   ostringstream oss;
   oss << "AvgLoss: " << (loss_-entropy_)/frames_ << " (Xent), "
      << "[AvgXent " << loss_/frames_ 
      << ", AvgTargetEnt " << entropy_/frames_ 
      << ", frames " << frames_ << "]" << endl;
   if (loss_vec_.size() > 0) {
      oss << "progress: [";
      copy(loss_vec_.begin(),loss_vec_.end(),ostream_iterator<float>(oss," "));
      oss << "]" << endl;
   }
   if (correct_ >= 0.0) {
      oss << "FRAME_ACCURACY >> " << 100.0*correct_/frames_ << "% <<" << endl;
   }
   return oss.str(); 
}

// -----------------------------------------------------------------------------------------------------

void LabelFrameLoss::Eval(const vector<uchar> &ref, const vector< vector<uchar> > &labels, const CuMatrixBase<BaseFloat> &nnet_out, MyCuMatrix<BaseFloat> *nnet_out_diff){
   int L = labels.size();
   int T = labels[0].size();

   assert(nnet_out.NumCols() == 2);
   assert(nnet_out.NumRows() % L == 0);
   assert(nnet_out_diff != NULL);

   Vector<BaseFloat> frm_weights(L * T);
   frm_weights.Set(1);

   vector< pair<int32, BaseFloat> > crr;
   crr.push_back(make_pair(0, 1.0));

   vector< pair<int32, BaseFloat> > incrr;
   incrr.push_back(make_pair(1, 1.0));

   Posterior targets(L * T);

#pragma omp parallel for
   for(int i = 0; i < L; ++i)
      for(int j = 0; j < T; ++j){
         targets[i + j*L] = (labels[0][j] == labels[i][j]) ? crr : incrr;
      }

   CuMatrix<BaseFloat> soft_out(L*T, 2);
   softmax.Propagate(nnet_out.RowRange(0, L*T), &soft_out);

   CuMatrix<BaseFloat> soft_diff;
   xent.Eval(frm_weights, soft_out, targets, &soft_diff);

   CuMatrix<BaseFloat> nnet_diff;
   softmax.Backpropagate(nnet_out.RowRange(0, L*T), soft_out,
         soft_diff, &nnet_diff);

   nnet_out_diff->Resize(nnet_out.NumRows(), nnet_out.NumCols());
   nnet_out_diff->RowRange(0, L*T).CopyFromMat(nnet_diff);
}

// -----------------------------------------------------------------------------------------------------

LabelMultiLoss::~LabelMultiLoss(){
   for(int i = 0; i < loss_arr_.size(); ++i)
      delete loss_arr_[i];
   loss_arr_.resize(0);
}

void LabelMultiLoss::Eval(const vector<uchar> &ref, const vector< vector<uchar> > &labels, const CuMatrixBase<BaseFloat> &nnet_out, MyCuMatrix<BaseFloat> *nnet_out_diff){

   if(nnet_out_diff != NULL)
      nnet_out_diff->Resize(nnet_out.NumRows(), nnet_out.NumCols());

   int colIdx = 0;
   for(int i = 0; i < loss_arr_.size(); ++i){
      
      int col_width = -1;

      switch(loss_arr_[i]->GetType()){
         case lList:
            col_width = 1;
            break;
         case lFrame:
            col_width = 2;
            break;
         default:
            assert(false);
      }

      MyCuMatrix<BaseFloat> diff;
      loss_arr_[i]->Eval(ref, labels, nnet_out.ColRange(colIdx, col_width), &diff);

      if(nnet_out_diff != NULL){
         diff.Scale(loss_weight_[i]);
         nnet_out_diff->ColRange(colIdx, col_width).CopyFromMat(diff);
      }

      colIdx += col_width;
   }
}

string LabelMultiLoss::Report(){
   string ret;
   for(int i = 0; i < loss_arr_.size(); ++i)
      ret += loss_arr_[i]->Report();
   return ret;
}
