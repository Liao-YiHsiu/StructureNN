#include "mynnet-loss.h"

const struct LabelLossBase::my_key_value LabelLossBase::myMarkerMap[] = {
   {LabelLossBase::lList,  "<LabelListLoss>"},
   {LabelLossBase::lFrame, "<LabelFrameLoss>"},
}

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
      string m(myMakrerMap[i].value);
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
   string conf_line, token;

   while(!is.eof()){
      assert(is.good());
      getline(is, conf_line);
      trim(conf_line);

      if(conf_line == "") continue;
      KALDI_VLOG(1) << conf_line;

      loss_arr.push_back(GetInstance(conf_line));
   }

   if(loss_arr.size() == 0)
      return NULL;
   else if(loss_arr.size() == 1)
      return loss_arr[0];
   else{
      return new MultiLoss(loss_arr);
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

LabelListLoss::~LabelListLoss(){
   if(strt_)
      delete strt_;
   strt_ = NULL;
}

void LabelListLoss::SetParam(istream &is){
   string strt_type, token;

   double sigma = 1.0, error = 0;
   ReadToken(is, false, &strt_type);

   while(!is.eof()){
      ReadToken(is, false, &token);
           if(token == "<Sigma>") ReadBasicType(is, false, &sigma);
      else if(token == "<Error>") ReadBasicType(is, false, &error);
      else KALDI_ERROR << "Unknown token " << token << ", a typo in config?"
         << " (Sigma|Error)";
      is >> std::ws;
   }

   strt_ = StrtListBase::getInstance(strt_type, sigma, error);
}

void LabelListLoss::Eval(const vector< vector<uchar> > &labels, const CuMatrixBase<BaseFloat> &nnet_out, CuMatrix<BaseFloat> *nnet_out_diff){

   int L = labels.size();
   int T = labels[0].size();

   assert(nnet_out.NumCols() == 1);
   assert(nnet_out.NumRows() % L == 0);
   assert(strt_ != NULL);

   Matrix<BaseFloat> nnet_out_host(nnet_out.NumRows(), nnet_out.NumCols(), kUndefined);
   nnet_out_host.CopyFromMat(nnet_out);

   vector<BaseFloat> targets(L);
   // sum up over all time frame
   Matrix<BaseFloat> nnet_out_mixed(L, 1);

#pragma omp parallel for
   for(int i = 0; i < L; ++i){
      targets[i] = phone_frame_acc(labels[0], labels[i]);

      for(int j = 0; j < T; ++j)
         nnet_out_mixed(i, 0) += nnet_out_host(i + j*L, 0);
   }

   Matrix<BaseFloat> nnet_diff_mixed;
   strt_->Eval(targets, nnet_out_mixed, nnet_diff_mixed);

   Matrix<BaseFloat> nnet_diff_host(nnet_out.NumRows(), nnet_out.NumCols());
   // dist error to all time frame
   for(int j = 0; j < T; ++j)
      for(int i = 0; i < L; ++i)
         nnet_diff_host(i + j*L, 0) = nnet_diff_mixed(i, 0);

   nnet_out_diff->CopyFromMat(nnet_diff_host);
}

// -----------------------------------------------------------------------------------------------------

void LabelFrameLoss:Eval(const vector< vector<uchar> > &labels, const CuMatrixBase<BaseFloat> &nnet_out, CuMatrix<BaseFloat> *nnet_out_diff){
   int L = labels.size();
   int T = labels[0].size();

   assert(nnet_out.NumCols() == 2);
   assert(nnet_out.NumRows() % L == 0);

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
         targets[i + j*L] = (label[0][j] == label[i][j]) ? crr : incrr;
      }

   CuMatrix<BaseFloat> nnet_diff;
   xent(frm_weights, nnet_out.RowRange(0, L * T), targets, &nnet_diff);

   if(nnet_out_diff != NULL)
      nnet_out_diff->RowRange(0, L*T).CopyFromMat(nnet_diff);
}

// -----------------------------------------------------------------------------------------------------

LabelMultiLoss::~LabelMultiLoss(){
   for(int i = 0; i < loss_arr_.size(); ++i)
      delete loss_arr_[i];
   loss_arr_.resize(0);
}

void LabelMultiLoss::Eval(const vector< vector<uchar> > &labels, const CuMatrixBase<BaseFloat> &nnet_out, CuMatrix<BaseFloat> *nnet_out_diff){

   if(nnet_out_diff != NULL)
      nnet_out_diff->Resize(nnet_out.NumRows(), nnet_out.NumCols());

   int colIdx = 0;
   for(int i = 0; i < loss_arr_.size(); ++i){
      
      int col_width = -1;

      switch(loss_arr[i]->GetType()){
         case lList:
            col_width = 1;
            break;
         case lFrame:
            col_width = 2;
            break;
         default:
            assert(false);
      }

      CuMatrix<BaseFloat> diff;
      loss_arr_[i]->Eval(labels, nnet_out.ColRange(colIdx, col_width), &diff);

      if(nnet_out_diff != NULL){
         nnet_out_diff.ColRange(colIdx, col_width).CopyFromMat(diff);
      }

      colIdx += col_width;
   }
}

string LabelMultiLoss::Report(){
   string ret;
   for(int i = 0; i < loss_arr_.size(); ++i)
      ret += loss_arr_[i].Report();
   return ret;
}
