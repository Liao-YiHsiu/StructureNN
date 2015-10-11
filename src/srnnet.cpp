#include "srnnet.h"

SRNnet::SRNnet(const SRNnet &other): nnet_transf_(other.nnet_transf_), 
   nnet1_(other.nnet1_), nnet2_(other.nnet2_), stateMax_(other.stateMax_),
   rnn_init_(other.rnn_init_) {

   mux_components_.resize(other.mux_components_.size());
   for(int i = 0; i < other.mux_components_.size(); ++i)
      mux_components_[i] = dynamic_cast<UpdatableComponent*>(other.mux_components_[i]->Copy());

   forw_component_ = dynamic_cast<UpdatableComponent*>(other.forw_component_->Copy());
   acti_component_ = other.acti_component_->Copy();

   Check();
}

SRNnet & SRNnet::operator = (const SRNnet &other) {
   Destroy();

   nnet_transf_ = other.nnet_transf_;
   nnet1_       = other.nnet1_;
   nnet2_       = other.nnet2_;
   stateMax_    = other.stateMax_;

   rnn_init_    = other.rnn_init_;

   mux_components_.resize(other.mux_components_.size());
   for(int i = 0; i < other.mux_components_.size(); ++i)
      mux_components_[i] = dynamic_cast<UpdatableComponent*>(other.mux_components_[i]->Copy());

   forw_component_ = dynamic_cast<UpdatableComponent*>(other.forw_component_->Copy());
   acti_component_ = other.acti_component_->Copy();

   Check();
   return *this;
}

SRNnet::~SRNnet(){
   Destroy();
}

void SRNnet::Propagate(const CuMatrix<BaseFloat> &in,
      const vector<vector<uchar>* > &labels, CuMatrix<BaseFloat> *out){
   int N = labels[0]->size();

   PropagateRPsi(in, labels);

   if(propagate_init_.NumRows() != propagate_frame_buf_[0].NumRows() ||
         propagate_init_.NumCols() != propagate_frame_buf_[0].NumCols())
      propagate_init_.Resize(propagate_frame_buf_[0].NumRows(),
            propagate_frame_buf_[0].NumCols(), kSetZero);

   for(int i = 0; i < N; ++i){
      if(i != 0)
         forw_component_->Propagate(propagate_acti_out_buf_[i-1], &propagate_forw_buf_[i]);
      else
         rnn_init_->Propagate(propagate_init_, &propagate_forw_buf_[i]);

      propagate_acti_in_buf_[i] = propagate_forw_buf_[i];
      propagate_acti_in_buf_[i].AddMat(1.0, propagate_frame_buf_[i]);

      acti_component_->Propagate(propagate_acti_in_buf_[i], &propagate_acti_out_buf_[i]);
   }

   nnet2_.Propagate(propagate_acti_out_buf_, propagate_score_buf_, N);
   
   Sum(propagate_score_buf_, out, N);
}

void SRNnet::Backpropagate(const CuMatrix<BaseFloat> &out_diff, 
      const vector<vector<uchar>* >&labels){

   //int L = labels.size();
   int N = labels[0]->size();

   if(backpropagate_acti_in_buf_.size() < N){
      backpropagate_acti_in_buf_.resize(N);
      backpropagate_acti_out_buf_.resize(N);
      backpropagate_forw_buf_.resize(N);
   }

   nnet2_.Backpropagate(out_diff, &backpropagate_acti_out_buf_, N);

   for(int i = N - 1; i >= 0; --i){
      if(i != N - 1)
         backpropagate_acti_out_buf_[i].AddMat(1.0, backpropagate_forw_buf_[i+1]);

      acti_component_->Backpropagate(propagate_acti_in_buf_[i], propagate_acti_out_buf_[i], 
            backpropagate_acti_out_buf_[i], &backpropagate_acti_in_buf_[i]);

      if(i != 0)
         forw_component_->Backpropagate(propagate_acti_out_buf_[i-1], propagate_forw_buf_[i],
               backpropagate_acti_in_buf_[i], &backpropagate_forw_buf_[i]);
      //else
      //   rnn_init_->BackPropagate(propagate_init_, propagate_forw_buf_[i],
      //         backpropagate_acti_in_buf_[i], &backpropagate_forw_buf_[i]);
   }

   if(backpropagate_phone_buf_.size() != mux_components_.size()){
      backpropagate_phone_buf_.resize(mux_components_.size());
      backpropagate_feat_buf_.resize(mux_components_.size());
   }

   BackRPsi(backpropagate_acti_in_buf_, labels, backpropagate_phone_buf_);

   for(int i = 0; i < mux_components_.size(); ++i)
      mux_components_[i]->Backpropagate(propagate_feat_buf_, propagate_phone_buf_[i],
            backpropagate_phone_buf_[i], &backpropagate_feat_buf_[i]);

   Sum(backpropagate_feat_buf_, &backpropagate_all_feat_buf_);

   nnet1_.Backpropagate(backpropagate_all_feat_buf_, NULL);
   
   //update Componnets
   for(int i = 0; i < mux_components_.size(); ++i)
      mux_components_[i]->Update(propagate_feat_buf_, backpropagate_phone_buf_[i]);

   rnn_init_->Update(propagate_init_, backpropagate_acti_in_buf_[0]);

   for(int i = 1; i < N; ++i)
      forw_component_->Update(propagate_acti_out_buf_[i-1], backpropagate_acti_in_buf_[i]);

}

void SRNnet::Propagate(const CuMatrix<BaseFloat> &in,
      const vector<uchar> &label, int pos, CuMatrix<BaseFloat> *out){
   int N = pos + 1;

   nnet_transf_.Feedforward(in.RowRange(0, pos+1), &transf_);

   nnet1_.Propagate(transf_, &propagate_feat_);

   // resize
   if(propagate_frame_buf_.size() < N){
      propagate_frame_buf_.resize(N);
      propagate_acti_in_buf_.resize(N);
      propagate_acti_out_buf_.resize(N);
      propagate_forw_buf_.resize(N);
      propagate_score_buf_.resize(N);
   }

   for(int i = 0; i < pos; ++i)
      mux_components_[label[i] - 1]->Propagate(
            propagate_feat_.RowRange(i, 1), &propagate_frame_buf_[i]);

   propagate_phone_.Resize(stateMax_, forw_component_->OutputDim(), kUndefined);

   for(int i = 0; i < mux_components_.size(); ++i){
      mux_components_[i]->Propagate(
            propagate_feat_.RowRange(pos, 1), &propagate_phone_tmp_);
      
      propagate_phone_.Row(i).CopyFromVec(propagate_phone_tmp_.Row(0));
   }

   if(propagate_init_.NumRows() != 1 ||
         propagate_init_.NumCols() != rnn_init_->InputDim())
      propagate_init_.Resize(1, rnn_init_->InputDim(), kSetZero);

   for(int i = 0; i < pos; ++i){
      if(i != 0)
         forw_component_->Propagate(propagate_acti_out_buf_[i-1], &propagate_forw_buf_[i]);
      else
         rnn_init_->Propagate(propagate_init_, &propagate_forw_buf_[i]);

      propagate_acti_in_buf_[i] = propagate_forw_buf_[i];
      propagate_acti_in_buf_[i].AddMat(1.0, propagate_frame_buf_[i]);

      acti_component_->Propagate(propagate_acti_in_buf_[i], &propagate_acti_out_buf_[i]);
   }

   if(pos != 0)
      forw_component_->Propagate(propagate_acti_out_buf_[pos-1], &propagate_forw_);
   else
      rnn_init_->Propagate(propagate_init_, &propagate_forw_);
   propagate_acti_in_ = propagate_phone_;
   propagate_acti_in_.AddVecToRows(1.0, propagate_forw_.Row(0), 1.0);

   acti_component_->Propagate(propagate_acti_in_, &propagate_acti_out_);

   nnet2_.Propagate(propagate_acti_out_, out);
}

void SRNnet::Backpropagate(const CuMatrix<BaseFloat> &out_diff, 
      const vector<uchar> &label, int pos, int depth){
   int N = pos+1;

   if(backpropagate_acti_out_buf_.size() < N){
      backpropagate_acti_out_buf_.resize(N);
      backpropagate_acti_in_buf_.resize(N);
      backpropagate_forw_buf_.resize(N);
   }

   nnet2_.Backpropagate(out_diff, &backpropagate_acti_out_);

   acti_component_->Backpropagate(propagate_acti_in_, propagate_acti_out_,
         backpropagate_acti_out_, &backpropagate_acti_in_);

   backpropagate_forw_.Resize(1, backpropagate_acti_in_.NumCols(), kSetZero);
   backpropagate_forw_.Row(0).AddRowSumMat(1.0, backpropagate_acti_in_, 1.0);

   if(pos != 0)
      forw_component_->Backpropagate(propagate_acti_out_buf_[pos-1], propagate_forw_,
            backpropagate_forw_, &backpropagate_acti_out_buf_[pos-1]);
   

   for(int i = pos - 1, d = 0; i >= 0 && d <= depth; --i, ++d){
      acti_component_->Backpropagate(propagate_acti_in_buf_[i], propagate_acti_out_buf_[i], 
            backpropagate_acti_out_buf_[i], &backpropagate_acti_in_buf_[i]);

      if(i != 0)
         forw_component_->Backpropagate(propagate_acti_out_buf_[i-1], propagate_forw_buf_[i],
               backpropagate_acti_in_buf_[i], &backpropagate_acti_out_buf_[i-1]);
   }

   backpropagate_feat_.Resize(stateMax_,   mux_components_[0]->InputDim(), kUndefined);
   backpropagate_all_feat_.Resize(pos + 1, mux_components_[0]->InputDim(), kSetZero);

   for(int i = 0; i < mux_components_.size(); ++i){
      mux_components_[i]->Backpropagate(
            propagate_feat_.RowRange(pos, 1), propagate_phone_.RowRange(i, 1),
            backpropagate_acti_in_.RowRange(i, 1), &backpropagate_feat_tmp_);
      backpropagate_feat_.Row(i).CopyFromVec(backpropagate_feat_tmp_.Row(0));
   }

   backpropagate_all_feat_.Row(pos).AddRowSumMat(1.0, backpropagate_feat_, 1.0);

   for(int i = pos - 1, d = 0; i >= 0 && d <= depth; --i, ++d){
      mux_components_[label[i] - 1]->Backpropagate(
            propagate_feat_.RowRange(i, 1), propagate_frame_buf_[i],
            backpropagate_acti_in_buf_[i], &backpropagate_all_feat_tmp_);
      backpropagate_all_feat_.Row(i).CopyFromVec(backpropagate_all_feat_tmp_.Row(0));
   }

   nnet1_.Backpropagate(backpropagate_all_feat_, NULL);

   //update Componnets
   for(int i = pos - 1, d = 0; i >= 0 && d <= depth; --i, ++d){
      mux_components_[label[i] - 1]->Update(
            propagate_feat_.RowRange(i, 1), backpropagate_acti_in_buf_[i]);
   }

   for(int i = 0; i < mux_components_.size(); ++i)
      mux_components_[i]->Update(propagate_feat_.RowRange(pos, 1), 
            backpropagate_acti_in_.RowRange(i, 1));

   for(int i = pos - 1, d = 0; i >= 0 && d <= depth; --i, ++d){
      if(i != 0)
         forw_component_->Update(propagate_acti_out_buf_[i-1],
               backpropagate_acti_in_buf_[i]);
      else
         rnn_init_->Update(propagate_init_, backpropagate_acti_in_buf_[0]);
   }

   if(pos != 0)
      forw_component_->Update(propagate_acti_out_buf_[pos-1], backpropagate_forw_);
   else
      rnn_init_->Update(propagate_init_, backpropagate_forw_);
}


void SRNnet::Decode(const CuMatrix<BaseFloat> &in, ScorePath::Table &table, int Nbest){
   int N = in.NumRows();

   vector< vector<uchar> >  dummy_label(stateMax_);
   vector< vector<uchar>* > dummy_label_ref(stateMax_);
   for(int i = 0; i < dummy_label.size(); ++i){
      dummy_label[i].resize(N, i + 1);
      dummy_label_ref[i] = &dummy_label[i];
   }

   PropagateRPsi(in, dummy_label_ref);

   // initialize double buffer
   if(double_buffer_.size() != 2){
      double_buffer_.resize(2);
      for(int i = 0; i < double_buffer_.size(); ++i){
         double_buffer_[i].resize(Nbest);
         for(int j = 0; j < Nbest; ++j){
            double_buffer_[i][j].Resize(stateMax_, forw_component_->OutputDim(), kUndefined);
         }
      }
   }

   // initialize backtrack & DP table
   vector< vector< Token > > DP_table(N);

   rnn_init_->Propagate(propagate_frame_buf_[0], &propagate_acti_in_);
   acti_component_->Propagate(propagate_acti_in_, &double_buffer_[0][0]);
   nnet2_.Propagate(double_buffer_[0][0], &propagate_score_);

   assert(propagate_score_.NumRows() == stateMax_);
   assert(propagate_score_.NumCols() == 1);

   propagate_score_host_.Resize(stateMax_, 1, kUndefined);
   propagate_score_host_.CopyFromMat(propagate_score_);

   DP_table[0].resize(stateMax_);
   for(int i = 0; i < stateMax_; ++i){
      DP_table[0][i].score = propagate_score_host_(i, 0);
      DP_table[0][i].phone = i+1;
      DP_table[0][i].bt    = 0;
   }
   sort(DP_table[0].begin(), DP_table[0].end(), compareToken);
   // initialization finished

   for(int i = 1; i < N; ++i){
      int tokenNum = DP_table[i - 1].size();
      if(tokenNum > Nbest) tokenNum = Nbest;

      DP_table[i].resize(tokenNum * stateMax_);

      for(int j = 0; j < tokenNum; ++j){
         int       bt    = DP_table[i - 1][j].bt;
         uchar     phone = DP_table[i - 1][j].phone;
         BaseFloat score = DP_table[i - 1][j].score;

         CuSubMatrix<BaseFloat> sub = 
            double_buffer_[(i - 1)%2][bt].RowRange(phone - 1, 1);

         forw_component_->Propagate(sub, &propagate_forw_);
         propagate_acti_in_ = propagate_frame_buf_[i];
         propagate_acti_in_.AddVecToRows(1.0, propagate_forw_.Row(0), 1.0);

         acti_component_->Propagate(propagate_acti_in_, &double_buffer_[i % 2][j]);
         nnet2_.Propagate(double_buffer_[i % 2][j], &propagate_score_);

         assert(propagate_score_.NumRows() == stateMax_);
         assert(propagate_score_.NumCols() == 1);

         // copy score back.
         propagate_score_host_.Resize(stateMax_, 1, kUndefined);
         propagate_score_host_.CopyFromMat(propagate_score_);

         for(int p = 0; p < stateMax_; ++p){
            DP_table[i][p + j * stateMax_].score = score + propagate_score_host_(p, 0);
            DP_table[i][p + j * stateMax_].phone = p+1;
            DP_table[i][p + j * stateMax_].bt    = j;
         }
      }

      // sorting for DP_table[i]
      sort(DP_table[i].begin(), DP_table[i].end(), compareToken);
   }

   int tokenNum = DP_table[N - 1].size();
   if(tokenNum > Nbest) tokenNum = Nbest;

   table.resize(tokenNum);
   for(int i = 0; i < tokenNum; ++i){
      table[i].first = DP_table[N - 1][i].score;
      table[i].second.resize(N);
      int index = i;
      for(int t = N - 1; t >= 0; --t){
         table[i].second[t] = DP_table[t][index].phone;
         index = DP_table[t][index].bt;
      }
   }
}

int32 SRNnet::InputDim() const{
   return nnet1_.InputDim();
}

int32 SRNnet::OutputDim() const{
   return nnet2_.OutputDim();
}

uchar SRNnet::StateMax() const{
   return stateMax_;
}

int32 SRNnet::NumParams() const{
   int32 n_params = 0;

   for(int i = 0; i < mux_components_.size(); ++i)
      n_params += mux_components_[i]->NumParams();

   n_params += rnn_init_->NumParams();
   n_params += forw_component_->NumParams();
   n_params += nnet1_.NumParams();
   n_params += nnet2_.NumParams();

   return n_params;
}

// TODO notimplement for forw_component_ & mux_components_ yet.
void SRNnet::SetDropoutRetention(BaseFloat r){
   nnet1_.SetDropoutRetention(r);
   nnet2_.SetDropoutRetention(r);
}

void SRNnet::Init(const Nnet& nnet1, const Nnet& nnet2, const string &config_file){
   nnet1_ = nnet1;
   nnet2_ = nnet2;

   Input in(config_file);
   istream &is = in.Stream();
   string conf_line, token;
   while(!is.eof()) {
      getline(is, conf_line);
      if(conf_line == "") continue;
      KALDI_VLOG(1) << conf_line;
      istringstream(conf_line) >> ws >> token;
      if( token == "<SRNnetProto>" || token == "</SRNnetProto>") continue;

      if( token == "<MUX>"){
         while(true){
            getline(is, conf_line);
            if(conf_line == "") continue;
            KALDI_VLOG(1) << conf_line;
            istringstream(conf_line) >> ws >> token;
            if(token == "</MUX>") break;
            mux_components_.push_back(
                  dynamic_cast<UpdatableComponent*>(Component::Init(conf_line + "\n")));
         }
         stateMax_ = mux_components_.size();

      }else{
         Component* comp = Component::Init(strAfter(conf_line, token) + "\n");

         if(token == "<Init>"){
            rnn_init_ = dynamic_cast<AddShift*>(comp);

         }else if(token == "<Forw>"){
            forw_component_ = dynamic_cast<UpdatableComponent*>(comp);

         }else if(token == "<Acti>"){
            acti_component_ = comp;

         }else{
            assert(false);
         }
      }
   }

   in.Close();
   Check();
}

void SRNnet::Read(const string &file){
   bool binary;
   Input in(file, &binary);
   Read(in.Stream(), binary);

   in.Close();
}

void SRNnet::Read(istream &is, bool binary){
   ExpectToken(is, binary, "<SRNnet>");

   nnet1_.Read(is, binary);
   nnet2_.Read(is, binary);

   int stateMax;
   ReadBasicType(is, binary, &stateMax);
   for(int i = 0; i < stateMax; ++i){
      mux_components_.push_back(dynamic_cast<UpdatableComponent*>(Component::Read(is, binary)));
   }
   stateMax_ = mux_components_.size();

   rnn_init_ = dynamic_cast<AddShift*>(Component::Read(is, binary));
   forw_component_ = dynamic_cast<UpdatableComponent*>(Component::Read(is, binary));
   acti_component_ = Component::Read(is, binary);
   
   ExpectToken(is, binary, "</SRNnet>");
   Check();
}

/// Write MLP to file
void SRNnet::Write(const string &file, bool binary) const{
   Output out(file, binary, true);
   Write(out.Stream(), binary);
   out.Close();
}

void SRNnet::Write(ostream &os, bool binary) const{
   Check();
   WriteToken(os, binary, "<SRNnet>");
   if(!binary) os << endl;

   nnet1_.Write(os, binary);
   nnet2_.Write(os, binary);

   int stateMax = mux_components_.size();
   WriteBasicType(os, binary, stateMax);
   for(int i = 0; i < mux_components_.size(); ++i)
      mux_components_[i]->Write(os, binary);

   rnn_init_->Write(os, binary);
   forw_component_->Write(os, binary);
   acti_component_->Write(os, binary);

   WriteToken(os, binary, "</SRNnet>");
   if(!binary) os << endl;
}

string SRNnet::Info() const{
  ostringstream ostr;
  ostr << "stateMax " << stateMax_ << endl;
  ostr << "input-dim " << InputDim() << endl;
  ostr << "output-dim " << OutputDim() << endl;
  ostr << "number-of-parameters(all) " << 
     static_cast<float>(NumParams())/1e6 << " millions" << endl;

  for (int32 i = 0; i < mux_components_.size(); i++) {
    ostr << "mux component " << i+1 << " : " 
         << Details(mux_components_[i]) << endl;
  }

  ostr << "Init input : " << Details(rnn_init_) << endl;
  ostr << "Forw component : " << Details(forw_component_) << endl;
  ostr << "Acti component : " << Details(acti_component_) << endl;

  ostr << "Nnet 1" << endl << nnet1_.Info() << endl;
  ostr << "Nnet 2" << endl << nnet2_.Info() << endl;

  return ostr.str();
}

string SRNnet::InfoGradient() const{
  ostringstream ostr;
  // gradient stats
  ostr << "### Gradient stats :\n";

  for (int32 i = 0; i < mux_components_.size(); i++) {
    ostr << "mux component " << i+1 << " : " 
         << Component::TypeToMarker(mux_components_[i]->GetType()) 
         << ", " << mux_components_[i]->InfoGradient() << endl;
  }

  ostr << "Init input : " 
         << Component::TypeToMarker(rnn_init_->GetType()) 
         << ", " << rnn_init_->InfoGradient() << endl;
  ostr << "Forw component : "
         << Component::TypeToMarker(forw_component_->GetType()) 
         << ", " << forw_component_->InfoGradient() << endl;
  ostr << "Acti component : " 
         << Component::TypeToMarker(acti_component_->GetType()) 
         << ", " << acti_component_->InfoGradient() << endl;

  ostr << "Nnet 1" << endl << nnet1_.InfoGradient() << endl;
  ostr << "Nnet 2" << endl << nnet2_.InfoGradient() << endl;
  return ostr.str();
}

string SRNnet::InfoPropagate() const{
  ostringstream ostr;
  //ostr << "### Forward propagation buffer content(phones) :\n";
  //ostr << "[0] output of <Input> " << MomentStatistics(propagate_feat_buf_) << endl;
  //for(int i = 0;i < mux_components_.size(); ++i)
  //   ostr << "[" << 1+i << "] output of <Mux> "
  //      << MomentStatistics(propagate_phone_buf_[i]) << endl;

  //ostr << "### Forward propagation buffer content(frames) :\n";
  //for(int i = 0; i < propagate_frame_buf_.size(); ++i)
  //   ostr << "[" << 1+i << "] output of <Frame> "
  //      << MomentStatistics(propagate_frame_buf_[i]) << endl
  //      << "[" << 1+i << "] input of <Acti> "
  //      << MomentStatistics(propagate_acti_in_buf_[i]) << endl
  //      << "[" << 1+i << "] output of <Acti> "
  //      << MomentStatistics(propagate_acti_out_buf_[i]) << endl
  //      << "[" << 1+i << "] output of <Forw> "
  //      << MomentStatistics(propagate_forw_buf_[i]) << endl
  //      << "[" << 1+i << "] output of <Score> "
  //      << MomentStatistics(propagate_score_buf_[i]) << endl;

  ostr << "Nnet 1" << endl << nnet1_.InfoPropagate() << endl;
  ostr << "Nnet 2" << endl << nnet2_.InfoPropagate() << endl;
  return ostr.str();
}

string SRNnet::InfoBackPropagate() const{
  ostringstream ostr;
  //ostr << "### Backward propagation buffer content(phones) :\n";
  //ostr << "[0] diff of <Input> " << MomentStatistics(backpropagate_all_feat_buf_) << endl;
  //for(int i = 0;i < mux_components_.size(); ++i)
  //   ostr << "[" << 1+i << "] diff of <Mux> "
  //      << MomentStatistics(backpropagate_phone_buf_[i]) << endl
  //      << "[" << 1+i << "] diff after <Mux> "
  //      << MomentStatistics(backpropagate_feat_buf_[i]) << endl;

  //ostr << "### Forward propagation buffer content(frames) :\n";
  //for(int i = 0; i < backpropagate_acti_out_buf_.size(); ++i)
  //   ostr << "[" << 1+i << "] diff of <Acti>"
  //      << MomentStatistics(backpropagate_acti_in_buf_[i]) << endl
  //      << "[" << 1+i << "] diff after <Acti>"
  //      << MomentStatistics(backpropagate_acti_out_buf_[i]) << endl
  //      << "[" << 1+i << "] diff of <Forw>"
  //      << MomentStatistics(backpropagate_forw_buf_[i]) << endl;

  ostr << "Nnet 1" << endl << nnet1_.InfoBackPropagate() << endl;
  ostr << "Nnet 2" << endl << nnet2_.InfoBackPropagate() << endl;
  return ostr.str();
}

void SRNnet::Check() const{
   KALDI_ASSERT(mux_components_.size() == stateMax_);
   
   for(int i = 1; i < mux_components_.size(); i++){
      KALDI_ASSERT(mux_components_[0]->OutputDim() == mux_components_[i]->OutputDim());
      KALDI_ASSERT(mux_components_[0]->InputDim()  == mux_components_[i]->InputDim());
   }

   KALDI_ASSERT(forw_component_->OutputDim() == forw_component_->InputDim());
   KALDI_ASSERT(rnn_init_->OutputDim() == rnn_init_->InputDim());
   KALDI_ASSERT(rnn_init_->OutputDim() == acti_component_->InputDim());
   KALDI_ASSERT(rnn_init_->OutputDim() == acti_component_->OutputDim());
   KALDI_ASSERT(rnn_init_->OutputDim() == forw_component_->OutputDim());
   KALDI_ASSERT(rnn_init_->OutputDim() == mux_components_[0]->OutputDim());

   KALDI_ASSERT(nnet1_.OutputDim() == mux_components_[0]->InputDim() );
   KALDI_ASSERT(nnet2_.InputDim()  == mux_components_[0]->OutputDim() );
}

void SRNnet::Destroy(){
   delete rnn_init_; rnn_init_ = NULL;
   delete forw_component_; forw_component_ = NULL;
   delete acti_component_; acti_component_ = NULL;

   for(int i = 0; i < mux_components_.size(); ++i)
      delete mux_components_[i];
   mux_components_.resize(0);

   propagate_phone_buf_.resize(0);
   propagate_frame_buf_.resize(0);
   propagate_acti_in_buf_.resize(0);
   propagate_acti_out_buf_.resize(0);
   propagate_forw_buf_.resize(0);
   propagate_score_buf_.resize(0);

   backpropagate_acti_out_buf_.resize(0);
   backpropagate_acti_in_buf_.resize(0);
   backpropagate_forw_buf_.resize(0);
   backpropagate_phone_buf_.resize(0);
   backpropagate_feat_buf_.resize(0);
}

void SRNnet::SetTrainOptions(const NnetTrainOptions& opts, double ratio){

   nnet1_.SetTrainOptions(opts);
   nnet2_.SetTrainOptions(opts);

   for(int i = 0; i < mux_components_.size(); ++i)
      mux_components_[i]->SetTrainOptions(opts);

   rnn_init_->SetTrainOptions(opts);
   forw_component_->SetTrainOptions(opts);
}

void SRNnet::SetTransform(const Nnet &nnet){
   nnet_transf_ = nnet;
}

void SRNnet::PropagateRPsi(const CuMatrix<BaseFloat> &in, const vector< vector<uchar>* > &labels){
   int N = labels[0]->size();

   nnet_transf_.Feedforward(in, &transf_);

   nnet1_.Propagate(transf_, &propagate_feat_buf_);
   
   if(propagate_phone_buf_.size() != mux_components_.size())
      propagate_phone_buf_.resize(mux_components_.size());

   for(int i = 0; i < mux_components_.size(); ++i)
      mux_components_[i]->Propagate(propagate_feat_buf_, &propagate_phone_buf_[i]);

   // resize
   if(propagate_frame_buf_.size() < N){
      propagate_frame_buf_.resize(N);
      propagate_acti_in_buf_.resize(N);
      propagate_acti_out_buf_.resize(N);
      propagate_forw_buf_.resize(N);
      propagate_score_buf_.resize(N);
   }

   RPsi(propagate_phone_buf_, labels, propagate_frame_buf_);
}

void SRNnet::RPsi(vector< CuMatrix<BaseFloat> > &propagate_phone,
      const vector< vector<uchar>* > &labels, vector< CuMatrix<BaseFloat> > &propagate_frame){
   
   int L = labels.size();
   int T = labels[0]->size();
   int D = propagate_phone[0].NumCols();

   KALDI_ASSERT(propagate_phone.size() == mux_components_.size());
   KALDI_ASSERT(propagate_frame.size() >= T);

   for(int i = 0; i < labels[0]->size(); ++i)
      propagate_frame[i].Resize(L, D, kUndefined);

   RPsiPack pack;
   packRPsi(propagate_phone, labels, propagate_frame, &pack);

   propRPsi(&pack);
}

void SRNnet::BackRPsi(vector< CuMatrix<BaseFloat> > &backpropagate_frame,
      const vector< vector<uchar>* > &labels, vector< CuMatrix<BaseFloat> > &backpropagate_phone){

   //int L = labels.size();
   int T = labels[0]->size();
   int D = backpropagate_frame[0].NumCols();

   KALDI_ASSERT(backpropagate_phone.size() == mux_components_.size());
   KALDI_ASSERT(backpropagate_frame.size() >= T);

   for(int i = 0; i < backpropagate_phone.size(); ++i)
      backpropagate_phone[i].Resize(T, D, kSetZero);

   RPsiPack pack;
   packRPsi(backpropagate_phone, labels, backpropagate_frame, &pack);

   backRPsi(&pack);
}

void SRNnet::packRPsi(vector< CuMatrix<BaseFloat> > &phone_mat,
      const vector< vector<uchar>* > &labels, vector< CuMatrix<BaseFloat> > &frame_mat, RPsiPack* pack){

   KALDI_ASSERT(pack != NULL);

   int L = labels.size();
   int T = labels[0]->size();
   int P = stateMax_;
   int D = forw_component_->InputDim();

   KALDI_ASSERT(phone_mat.size() == P);
   KALDI_ASSERT(frame_mat.size() >= T);

   labelbuf_.resize(L*T);

   for(int i = 0; i < L; i++)
      for(int j = 0; j < T; j++)
         labelbuf_[i * T + j] = (*labels[i])[j];

   labelbuf_dev_.Resize(L, T);
   labelbuf_dev_.CopyFromVec(labelbuf_);


   int phone_mat_stride = phone_mat[0].Stride();
   phone_mat_pt_.resize(P);
   for(int i = 0; i < P; ++i){
      KALDI_ASSERT(phone_mat_stride == phone_mat[i].Stride());
      phone_mat_pt_[i] = getCuPointer(&phone_mat[i]);
   }

   phone_mat_pt_dev_.Resize(P);
   phone_mat_pt_dev_.CopyFromVec(phone_mat_pt_);


   int frame_mat_stride = frame_mat[0].Stride();
   frame_mat_pt_.resize(T);
   for(int i = 0; i < T; ++i){
      KALDI_ASSERT(frame_mat_stride == frame_mat[i].Stride());
      frame_mat_pt_[i] = getCuPointer(&frame_mat[i]);
   }

   frame_mat_pt_dev_.Resize(T);
   frame_mat_pt_dev_.CopyFromVec(frame_mat_pt_);

   pack->L                 = L;
   pack->T                 = T;
   pack->P                 = P;
   pack->D                 = D;
   pack->phone_feat_stride = phone_mat_stride;
   pack->frame_feat_stride = frame_mat_stride;
   pack->lab               = labelbuf_dev_.Data();
   pack->phone_feat        = phone_mat_pt_dev_.Data();
   pack->frame_feat        = frame_mat_pt_dev_.Data();
}

string SRNnet::Details(const Component* comp) const{
   ostringstream os;

   os << Component::TypeToMarker(comp->GetType())
      << ", input-dim " << comp->InputDim()
      << ", output-dim " << comp->OutputDim()
      << ", " << comp->Info();

   return os.str();
}

bool compareToken(const Token &a, const Token &b){ return a.score > b.score; }
