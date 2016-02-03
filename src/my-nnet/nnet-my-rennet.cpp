#include "my-nnet/nnet-my-rennet.h"

NnetBatch::NnetBatch(const NnetBatch& other): MyNnet(other){
   SetBatchNum(other.batch_num_);
   Check();
}

NnetBatch& NnetBatch::operator= (const NnetBatch &other){
   MyNnet::operator=(other);
   Destroy();

   SetBatchNum(other.batch_num_);
   Check();

   return *this;
}

void NnetBatch::Propagate(int index, const CuMatrixBase<BaseFloat> &in, MyCuMatrix<BaseFloat> *out){
   assert( index < batch_num_ );
   assert( out != NULL );

   assert( propagate_buf_arr_[index].size() >= NumComponents() - 1);

   if( NumComponents() == 0 ){
      (*out) = in;
      return;
   }else if(NumComponents() == 1){
      components_[0]->Propagate(in, out);
   }else{
      components_[0]->Propagate(in, &propagate_buf_arr_[index][0]);

      int last = NumComponents() - 1;
      for(int i = 1; i < last; ++i)
         components_[i]->Propagate(propagate_buf_arr_[index][i - 1], &propagate_buf_arr_[index][i]);

      components_[last]->Propagate(propagate_buf_arr_[index][last - 1], out);
   }
}

void NnetBatch::Backpropagate(int index,
      const CuMatrixBase<BaseFloat> &in, const CuMatrixBase<BaseFloat> &out, 
      const CuMatrixBase<BaseFloat> &out_diff, MyCuMatrix<BaseFloat> *in_diff){

   assert( index < batch_num_ );

   assert(propagate_buf_arr_[index].size() >= NumComponents() - 1);
   assert(backpropagate_buf_arr_[index].size() >= NumComponents() - 1);

   if( NumComponents() == 0){
      (*in_diff) = out_diff;
      return;
   }else if(NumComponents() == 1){
      components_[0]->Backpropagate(in, out, out_diff, in_diff);
      
   }else{
      int last = NumComponents() - 1;
      components_[last]->Backpropagate(propagate_buf_arr_[index][last - 1], out, 
            out_diff, &backpropagate_buf_arr_[index][last -1]);

      for(int i = last - 1; i >= 1; --i){
         components_[i]->Backpropagate(propagate_buf_arr_[index][i - 1], propagate_buf_arr_[index][i],
               backpropagate_buf_arr_[index][i], &backpropagate_buf_arr_[index][i - 1]);
      }

      components_[0]->Backpropagate(in, propagate_buf_arr_[index][0], 
            backpropagate_buf_arr_[index][0], in_diff);
   }
}

string NnetBatch::InfoPropagate(int index) const{
   ostringstream ostr;

   ostr << "### Forward propagation buffer content :\n";
   for( int i = 0; i < NumComponents()-1; ++i){
      ostr << "["<<i+1<<"] output of "
         << MyComponent::TypeToMarker(components_[i]->GetType())
         << MomentStatistics(propagate_buf_arr_[index][i]) << endl;
   }

   return ostr.str();
}

string NnetBatch::InfoBackPropagate(int index) const {
   ostringstream ostr;

   ostr << "### Backward propagation buffer content :\n";
   for(int i = 0; i < NumComponents() - 1; ++i){
      ostr << "["<<i+1<<"] diff-output of "
         << MyComponent::TypeToMarker(components_[i]->GetType())
         << MomentStatistics(backpropagate_buf_arr_[index][i]) << endl;
   }

   return ostr.str();
}

void NnetBatch::SetBatchNum(int32 batch_num){
   batch_num_ = batch_num;

   propagate_buf_arr_.resize(batch_num_);
   backpropagate_buf_arr_.resize(batch_num_);

   int buff_num = NumComponents() - 1;
   if(buff_num < 0) buff_num = 0;

   for(int i = 0; i < batch_num_; ++i){
      propagate_buf_arr_[i].resize(buff_num);
      backpropagate_buf_arr_[i].resize(buff_num);
   }
}

void NnetBatch::Check() const{
   MyNnet::Check();

   assert(batch_num_ == propagate_buf_arr_.size());
   assert(batch_num_ == backpropagate_buf_arr_.size());
   for(int i = 0; i < batch_num_; ++i){
      assert(propagate_buf_arr_[i].size() >= NumComponents() - 1 );
      assert(backpropagate_buf_arr_[i].size() >= NumComponents() - 1);
   }
}

void NnetBatch::Destroy(){
   propagate_buf_arr_.resize(0);
   backpropagate_buf_arr_.resize(0);

   batch_num_ = 0;
}


RENnet::RENnet(const RENnet& other){
   sub_models_.resize(other.sub_models_.size());
   for(int i = 0; i < sub_models_.size(); ++i)
      sub_models_[i] = other.sub_models_[i]->Copy();

   sel_model_ = other.sel_model_->Copy();
   nnet_in_   = new MyNnet(*other.nnet_in_);
   nnet_out_  = new MyNnet(*other.nnet_out_);
   att_model_ = other.att_model_->Copy();

   depth_ = other.depth_;
   opts_  = other.opts_;
}

RENnet& RENnet::operator=(const RENnet &other){
   Destroy();
   
   sub_models_.resize(other.sub_models_.size());
   for(int i = 0; i < sub_models_.size(); ++i)
      sub_models_[i] = other.sub_models_[i]->Copy();

   sel_model_ = other.sel_model_->Copy();
   nnet_in_   = new MyNnet(*other.nnet_in_);
   nnet_out_  = new MyNnet(*other.nnet_out_);
   att_model_ = other.att_model_->Copy();

   depth_ = other.depth_;
   opts_  = other.opts_;

   return *this;
}

void RENnet::Propagate(const CuMatrixBase<BaseFloat> &in, MyCuMatrix<BaseFloat> *out){
   assert(out != NULL);
   CheckBuff();
   
   if(nnet_in_ != NULL){
      nnet_in_->Propagate(in, &propagate_buf_[0]);
   }else{
      propagate_buf_[0] = in;
   }

   for(int i = 0; i < depth_; ++i){

      for(int j = 0; j < sub_models_.size(); ++j){
         sub_models_[j]->Propagate(i, propagate_buf_[i], &propagate_buf_sub_[i][j]);
      }

      sel_model_->Propagate(i, propagate_buf_[i], &propagate_buf_att_[i]);

      att_model_->Propagate(propagate_buf_sub_[i], propagate_buf_att_[i], &propagate_buf_[i+1]);
   }


   if(nnet_out_ != NULL){
      nnet_out_->Propagate(propagate_buf_[depth_], out);
   }else{
      (*out) = propagate_buf_[depth_];
   }
}

void RENnet::Backpropagate(const CuMatrixBase<BaseFloat> &out_diff, MyCuMatrix<BaseFloat> *in_diff){
   CheckBuff();

   if(nnet_out_ != NULL){
      nnet_out_->Backpropagate(out_diff, &backpropagate_buf_[depth_]);
   }else{
      backpropagate_buf_[depth_] = out_diff;
   }

   for(int i = depth_-1; i >= 0; --i){
      att_model_->Backpropagate(propagate_buf_sub_[i], propagate_buf_att_[i], 
            backpropagate_buf_[i+1], backpropagate_buf_sub_[i], backpropagate_buf_att_[i]);

      sel_model_->Backpropagate(i, propagate_buf_[i], propagate_buf_att_[i],
            backpropagate_buf_att_[i], &backpropagate_buf_[i]);

      for(int j = 0; j < sub_models_.size(); ++j){
         sub_models_[j]->Backpropagate(i, propagate_buf_[i], propagate_buf_sub_[i][j],
               backpropagate_buf_sub_[i][j], &tmp_cumat_);
         backpropagate_buf_[i].AddMat(1.0, tmp_cumat_);
      }
   }

   if(nnet_in_ != NULL){
      nnet_in_->Backpropagate(backpropagate_buf_[0], in_diff);
   }else{
      if(in_diff != NULL)
         (*in_diff) = backpropagate_buf_[0];
   }
}

void RENnet::Update(){
   if(nnet_in_) nnet_in_->Update();
   if(nnet_out_) nnet_out_->Update();

   for(int i = 0; i < sub_models_.size(); ++i)
      sub_models_[i]->Update();

   sel_model_->Update();
}

void RENnet::Feedforward(const CuMatrixBase<BaseFloat> &in, MyCuMatrix<BaseFloat> *out){
   assert(out != NULL);
   CheckBuff();
   
   if(nnet_in_ != NULL){
      nnet_in_->Propagate(in, &propagate_buf_[0]);
   }else{
      propagate_buf_[0] = in;
   }

   for(int i = 0; i < depth_; ++i){

      int now = i % 2;
      int next = (i+1) % 2;

      for(int j = 0; j < sub_models_.size(); ++j){
         sub_models_[j]->Propagate(0, propagate_buf_[now], &propagate_buf_sub_[now][j]);
      }

      sel_model_->Propagate(0, propagate_buf_[now], &propagate_buf_att_[now]);

      att_model_->Propagate(propagate_buf_sub_[now], propagate_buf_att_[now], &propagate_buf_[next]);
   }

   if(nnet_out_ != NULL){
      nnet_out_->Propagate(propagate_buf_[depth_ % 2], out);
   }else{
      (*out) = propagate_buf_[depth_ % 2];
   }
}

int32 RENnet::InputDim() const{
   assert(!sub_models_.empty());
   if(nnet_in_ != NULL) return nnet_in_->InputDim();
   return sel_model_->InputDim();
}

int32 RENnet::OutputDim() const{
   assert(!sub_models_.empty());
   if(nnet_out_ != NULL) return nnet_out_->OutputDim();
   return sel_model_->InputDim(); // output dim == input dim for recursive layer
}

const NnetBatch& RENnet::GetSubModel(int32 i) const{
   assert(i < sub_models_.size());
   return *(sub_models_[i]);
}

NnetBatch& RENnet::GetSubModel(int32 i){
   assert(i < sub_models_.size());
   return *(sub_models_[i]);
}

void RENnet::SetSubModel(int32 i, NnetBatch *nnet_batch){
   assert(i < sub_models_.size());
   delete sub_models_[i];
   sub_models_[i] = nnet_batch;
   Check();
}

void RENnet::SetSelModel(NnetBatch *nnet_batch){
   delete sel_model_;
   sel_model_ = nnet_batch;
   Check();
}

int32 RENnet::NumParams() const{
   int sum = 0;
   for(int i = 0; i < sub_models_.size(); ++i){
      sum += sub_models_[i]->NumParams();
   }

   sum += sel_model_->NumParams();
   sum += nnet_in_  ->NumParams();
   sum += nnet_out_ ->NumParams();

   return sum;
}

void RENnet::GetParams(Vector<BaseFloat> *weight) const{
   weight->Resize(NumParams());
   int pos = 0;
   for(int i = 0; i < sub_models_.size(); ++i){
      Vector<BaseFloat> c_params;
      sub_models_[i]->GetParams(&c_params);
      weight->Range(pos, c_params.Dim()).CopyFromVec(c_params);
      pos += c_params.Dim();
   }

   assert(pos == NumParams());
}

void RENnet::SetDropoutRetention(BaseFloat r){
   if(nnet_in_)  nnet_in_->SetDropoutRetention(r);
   if(nnet_out_) nnet_out_->SetDropoutRetention(r);

   for(int i = 0; i < sub_models_.size(); ++i){
      sub_models_[i]->SetDropoutRetention(r);
   }

   sel_model_->SetDropoutRetention(r);

}

void RENnet::Init(istream &is){
   ExpectToken(is, false, "<RENnet>");
   string conf_line, token;

   while(!is.eof()){
      assert(is.good());
      getline(is, conf_line);
      if(conf_line == "") continue;
      istringstream(conf_line) >> ws >> token;

      if(token == "<RENnet-sub>"){
         int32 num_sub_models;
         ReadBasicType(is, false, &num_sub_models);

         for(int i = 0; i < num_sub_models; ++i){
            NnetBatch* nnet_batch = new NnetBatch();
            nnet_batch->Init(is);
            sub_models_.push_back(nnet_batch);
         }
      }else if(token == "<RENnet-sel>"){
         sel_model_ = new NnetBatch();
         sel_model_->Init(is);
      }else if(token == "<Nnet-in>"){
         nnet_in_ = new MyNnet();
         nnet_in_->Init(is);
      }else if(token == "<Nnet-out>"){
         nnet_out_   = new MyNnet();
         nnet_out_->Init(is);
      }else if(token == "</RENnet>"){
         break;
      }else
         assert(false);
   }

   att_model_ = new AttComponent();

   SetDepth(0);
   Check();
}

void RENnet::Init(const string& config_file){
   Input in(config_file);
   istream &is = in.Stream();

   Init(is);

   in.Close();
   Check();
}

void RENnet::Read(const string& file){
   bool binary;
   Input in(file, &binary);
   Read(in.Stream(), binary);
   in.Close();
}

void RENnet::Read(istream &is, bool binary){
   ExpectToken(is, binary, "<RENnet>");
   
   int32 num_sub_models;
   ReadBasicType(is, binary, &num_sub_models);
   sub_models_.resize(num_sub_models);
   for(int i = 0; i < num_sub_models; ++i){
      sub_models_[i] = new NnetBatch();
      sub_models_[i]->Read(is, binary);
   }

   sel_model_ = new NnetBatch();
   sel_model_->Read(is, binary);

   nnet_in_   = new MyNnet();
   nnet_in_  ->Read(is, binary);

   nnet_out_  = new MyNnet();
   nnet_out_ ->Read(is, binary);

   att_model_ = new AttComponent();

   ExpectToken(is, binary, "</RENnet>");

   SetDepth(0);
   Check();
}

void RENnet::Write(const string &file, bool binary) const {
   Output out(file, binary, true);
   Write(out.Stream(), binary);
   out.Close();
}

void RENnet::Write(ostream &os, bool binary) const {
   Check();
   WriteToken(os, binary, "<RENnet>");
   if(!binary) os << endl;

   WriteBasicType(os, binary, (int32)sub_models_.size());
   for(int i = 0; i < sub_models_.size(); ++i){
      sub_models_[i]->Write(os, binary);
   }

   sel_model_->Write(os, binary);
   nnet_in_  ->Write(os, binary);
   nnet_out_ ->Write(os, binary);

   WriteToken(os, binary, "</RENnet>");
   if(!binary) os << endl;
}

string RENnet::Info() const{
   ostringstream ostr;
   ostr << "num-sub-model " << NumSubModels() << endl;
   ostr << "input-dim " << InputDim() << endl;
   ostr << "output-dim " << OutputDim() << endl;
   ostr << "total number-of-parameters " << NumParams()/(float)1e6 
      << " millions" << endl;

   ostr << "====================NNET IN======================" << endl;
   ostr << nnet_in_->Info();

   ostr << "====================SUB MODEL====================" << endl;
   for(int i = 0; i < sub_models_.size(); ++i){
      ostr << sub_models_[i]->Info();
   }
   ostr << "====================SEL MODEL====================" << endl;
   ostr << sel_model_->Info();

   ostr << "====================NNET OUT=====================" << endl;
   ostr << nnet_out_->Info();

   return ostr.str();
}

string RENnet::InfoGradient() const{
   ostringstream ostr;

   ostr << "### Gradient stats :\n";

   ostr << "====================NNET IN======================" << endl;
   ostr << nnet_in_->InfoGradient();

   ostr << "====================SUB MODEL====================" << endl;
   for(int i = 0; i < sub_models_.size(); ++i){
      ostr << sub_models_[i]->InfoGradient();
   }
   ostr << "====================SEL MODEL====================" << endl;
   ostr << sel_model_->InfoGradient();

   ostr << "====================NNET OUT=====================" << endl;
   ostr << nnet_out_->InfoGradient();

   return ostr.str();
}

string RENnet::InfoPropagate() const{
   ostringstream ostr;

   ostr << "====================NNET IN======================" << endl;
   ostr << nnet_in_->InfoPropagate();

   ostr << "[0] output of <Nnet-In> " << MomentStatistics(propagate_buf_[0]) << endl;

   for(int i = 0; i < depth_; ++i){
      ostr << "====================SUB MODEL====================" << endl;
      for(int j = 0; j < sub_models_.size(); ++j){
         ostr << sub_models_[j]->InfoPropagate(i);
         ostr << "[" << i << "] output of sub-model[" << j << "] " 
            << MomentStatistics(propagate_buf_sub_[i][j]) << endl;
      }
      ostr << "====================SEL MODEL====================" << endl;
      ostr << sel_model_->InfoPropagate(i);
      ostr << "[" << i << "] output of sel-model " << MomentStatistics(propagate_buf_att_[i]) << endl;

      ostr << "[" << i+1 << "] output of <Recurrent> "  << MomentStatistics(propagate_buf_[i + 1]) << endl;
   }

   ostr << "====================NNET OUT=====================" << endl;
   ostr << nnet_out_->InfoPropagate();

   return ostr.str();
}

string RENnet::InfoBackPropagate() const{
   ostringstream ostr;

   ostr << "====================NNET IN======================" << endl;
   ostr << nnet_in_->InfoBackPropagate();

   ostr << "[0] diff-output of <Nnet-In> " << MomentStatistics(backpropagate_buf_[0]) << endl;

   for(int i = 0; i < depth_; ++i){
      ostr << "====================SUB MODEL====================" << endl;
      for(int j = 0; j < sub_models_.size(); ++j){
         ostr << sub_models_[j]->InfoBackPropagate(i);
         ostr << "[" << i << "] diff-output of sub-model[" << j << "] " 
            << MomentStatistics(backpropagate_buf_sub_[i][j]) << endl;
      }
      ostr << "====================SEL MODEL====================" << endl;
      ostr << sel_model_->InfoBackPropagate(i);
      ostr << "[" << i << "] diff-output of sel-model " <<
         MomentStatistics(backpropagate_buf_att_[i]) << endl;

      ostr << "[" << i+1 << "] diff-output of <Recurrent> "  << 
         MomentStatistics(backpropagate_buf_[i + 1]) << endl;
   }

   ostr << "====================NNET OUT=====================" << endl;
   ostr << nnet_out_->InfoBackPropagate();

   return ostr.str();
}

void RENnet::SetDepth(int32 depth){
   assert(depth >= 0);
   depth_ = depth;
   for(int i = 0; i < sub_models_.size(); ++i){
      sub_models_[i]->SetBatchNum(depth);
   }
   sel_model_->SetBatchNum(depth);

   // SetBuff()
}

void RENnet::Check() const{
   int num_sub_models = sub_models_.size();
   assert(num_sub_models > 0);
   assert(sel_model_ != NULL);
   assert(att_model_ != NULL);

   for(int i = 0; i < num_sub_models; ++i){
      assert(sub_models_[i] != NULL);
   }

   int rlayer = sub_models_[0]->InputDim();

   for(int i = 0; i < num_sub_models; ++i){
      assert(rlayer == sub_models_[i]->InputDim());
      assert(rlayer == sub_models_[i]->OutputDim());
   }
   assert(rlayer == sel_model_->InputDim());
   assert(num_sub_models == sel_model_->OutputDim());

   if(nnet_in_ != NULL){
      assert(nnet_in_->OutputDim() == rlayer);
   }
   
   if(nnet_out_ != NULL){
      assert(nnet_out_->InputDim() == rlayer);
   }

   CheckBuff();
}

void RENnet::Destroy(){
   if(nnet_in_) delete nnet_in_;
   nnet_in_ = NULL;
   if(nnet_out_) delete nnet_out_;
   nnet_out_ = NULL;

   delete att_model_; att_model_ = NULL;
   delete sel_model_; sel_model_ = NULL;

   for(int i = 0; i < sub_models_.size(); ++i){
      delete sub_models_[i];
   }
   sub_models_.resize(0);

   propagate_buf_.resize(0);
   propagate_buf_att_.resize(0);
   propagate_buf_sub_.resize(0);

   backpropagate_buf_.resize(0);
   backpropagate_buf_att_.resize(0);
   backpropagate_buf_sub_.resize(0);
}

void RENnet::SetTrainOptions(const NnetTrainOptions& opts){
   if(nnet_in_)  nnet_in_->SetTrainOptions(opts);
   if(nnet_out_) nnet_out_->SetTrainOptions(opts);
   for(int i = 0; i < sub_models_.size(); ++i)
      sub_models_[i]->SetTrainOptions(opts);
   sel_model_->SetTrainOptions(opts);
}


void RENnet::CheckBuff()const{
   assert(propagate_buf_.size() == depth_ + 1);
   assert(propagate_buf_att_.size() == depth_);
   assert(propagate_buf_sub_.size() == depth_);
   for(int i = 0; i < propagate_buf_sub_.size(); ++i){
      assert(propagate_buf_sub_[i].size() == sub_models_.size());
   }

   assert(backpropagate_buf_.size() == depth_ + 1);
   assert(backpropagate_buf_att_.size() == depth_);
   assert(backpropagate_buf_sub_.size() == depth_);
   for(int i = 0; i < backpropagate_buf_sub_.size(); ++i){
      assert(backpropagate_buf_sub_[i].size() == sub_models_.size());
   }

}
