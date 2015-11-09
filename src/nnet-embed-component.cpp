#include "nnet-embed-component.h"

void Embed::Propagate(const CuMatrixBase<BaseFloat> &in, 
      CuMatrix<BaseFloat> *out){
   assert(input_dim_ == in.NumCols());
   assert(in.NumRows() * seq_stride_ == seq_.size());

   out->Resize(seq_.size(), output_dim_, kSetZero);

   PropagateFnc(in, out);
}

void Embed::Backpropagate(const CuMatrixBase<BaseFloat> &in,
      const CuMatrixBase<BaseFloat> &out,
      const CuMatrixBase<BaseFloat> &out_diff,
      CuMatrix<BaseFloat> *in_diff){

   assert(in.NumRows()*seq_stride_ == seq_.size() && in.NumCols() == input_dim_);
   assert(out.NumRows() == seq_.size() && out.NumCols() == output_dim_);
   assert(out_diff.NumRows() == seq_.size() && out_diff.NumCols() == output_dim_);

   if(in_diff != NULL)
      in_diff->Resize(in.NumRows(), input_dim_, kSetZero);

   BackpropagateFnc(in, out, out_diff, in_diff);
}

void Embed::InitData(istream &is){
   int32 label_num;
   ExpectToken(is, false, "<LabelNum>");
   ReadBasicType(is, false, &label_num);

   SetLabelNum(label_num);
}

void Embed::ReadData(istream &is, bool binary){
   int32 label_num;
   ExpectToken(is, binary, "<LabelNum>");
   ReadBasicType(is, binary, &label_num);

   SetLabelNum(label_num);
}

void Embed::WriteData(ostream &os, bool binary) const {
   WriteToken(os, binary, "<LabelNum>");
   WriteBasicType(os, binary, label_num_);
}

void Embed::SetLabelSeqs(const vector<int32> &seq, int seq_stride){
   assert(seq.size() % seq_stride == 0);
   for(int i = 0; i < seq.size(); ++i)
      assert(seq[i] < label_num_);

   seq_stride_ = seq_stride;
   seq_ = seq;

   SetLabelSeqsFnc(seq, seq_stride);
}

void Embed::SetLabelNum(int label_num) { 
   assert(label_num > 0);
   label_num_ = label_num; 
   SetLabelNumFnc(label_num);
}

// --------------------------------------------------------------------------------------------

Component* EmbedSimple::Copy() const{
   EmbedSimple *m = new EmbedSimple(input_dim_, output_dim_);
   m->SetLabelNum(GetLabelNum());

   return m;
}

void EmbedSimple::PropagateFnc(const CuMatrixBase<BaseFloat> &in,
      CuMatrixBase<BaseFloat> *out){
   
   embed_prop(in, seq_device_.Data(), seq_stride_, *out);

   // check consistence
   //CuMatrix<BaseFloat> tmp_out(seq_.size(), output_dim_, kUndefined);
   //Matrix<BaseFloat>   tmp_sub(seq_.size(), label_num_, kSetZero);

   //for(int i = 0; i*seq_stride_ < seq_.size(); ++i){
   //   CuSubMatrix<BaseFloat> sub(tmp_out, i*seq_stride_, seq_stride_, 0, in.NumCols());
   //   sub.CopyRowsFromVec(in.Row(i));
   //}

   //for(int i = 0; i < seq_.size(); ++i)
   //   tmp_sub(i, seq_[i]) = 1;

   //CuSubMatrix<BaseFloat> sub(tmp_out, 0, seq_.size(), in.NumCols(), label_num_);
   //sub.CopyFromMat(tmp_sub);

   //assert(Same(*out, tmp_out));

}

void EmbedSimple::BackpropagateFnc(const CuMatrixBase<BaseFloat> &in,
      const CuMatrixBase<BaseFloat> &out,
      const CuMatrixBase<BaseFloat> &out_diff,
      CuMatrixBase<BaseFloat> *in_diff){

   if(in_diff == NULL) return;

   embed_back(out_diff, seq_stride_, *in_diff);

   // check consistence
   //CuSubMatrix<BaseFloat> real_diff(out_diff, 0, seq_.size(), 0, in.NumCols());
   //CuMatrix<BaseFloat> tmp_in_diff(in.NumRows(), input_dim_, kSetZero);

   //for(int i = 0; i < seq_.size(); ++i){
   //   tmp_in_diff.Row(i/seq_stride_).AddVec(1.0, real_diff.Row(i), 1.0);
   //}

   //assert(Same(*in_diff, tmp_in_diff));
}

void EmbedSimple::SetLabelSeqsFnc(const vector<int32> &seq, int seq_stride){
   seq_device_ = seq;
}

void EmbedSimple::SetLabelNumFnc(int label_num){
   assert(input_dim_ + label_num == output_dim_);
}

// ----------------------------------------------------------------------------------------------
//
EmbedMux::~EmbedMux(){
   for(int i = 0; i < comps_.size(); ++i)
      if(comps_[i]) delete comps_[i];
   comps_.resize(0);
}

Component* EmbedMux::Copy() const{
   EmbedMux *m = new EmbedMux(input_dim_, output_dim_);
   m->SetLabelNum(GetLabelNum());

   for(int i = 0; i < comps_.size(); ++i)
      m->comps_[i] = comps_[i]->Copy();

   return m;
}

string EmbedMux::Info() const{
   ostringstream os;
   for(int i = 0; i < comps_.size(); ++i)
      os << "Component " << (i+1) << ": \n" << comps_[i]->Info();

   return os.str();
}

string EmbedMux::InfoGradient() const{
   ostringstream os;
   for(int i = 0; i < comps_.size(); ++i)
      os << "Component " << (i+1) << ": \n" << comps_[i]->InfoGradient();

   return os.str();
}

void EmbedMux::PropagateFnc(const CuMatrixBase<BaseFloat> &in, CuMatrixBase<BaseFloat> *out){

   // clear all previous buf...
   for(int i = 0; i < in_buff_.size(); ++i){
      in_buff_[i].Resize(0, 0, kUndefined);
      in_diff_buff_[i].Resize(0, 0, kUndefined);
      out_buff_[i].Resize(0, 0, kUndefined);
      out_diff_buff_[i].Resize(0, 0, kUndefined);
   }

   for(int i = 0; i < cnt_.size(); ++i){
      if(cnt_[i] != 0){
         in_buff_[i].Resize(cnt_[i], input_dim_, kUndefined);
      }
   }

   dist_prop(in, seq_device_.Data(), seq_stride_, id_device_.Data(), getVecCuMatrixPt(in_buff_), getVecCuMatrixStride(in_buff_));

   for(int i = 0; i < comps_.size(); ++i)
      if(cnt_[i] != 0){
         comps_[i]->Propagate(in_buff_[i], &out_buff_[i]);
      }

   comb_prop(getVecCuMatrixPt(out_buff_), getVecCuMatrixStride(out_buff_), seq_device_.Data(), seq_stride_, id_device_.Data(), *out);

   // check consistence
   //CuMatrix<BaseFloat> tmp_out(seq_.size(), output_dim_, kUndefined);
   //CuMatrix<BaseFloat> tmp_row(1, output_dim_, kUndefined);
   //for(int i = 0; i < seq_.size(); ++i){
   //   comps_[seq_[i]]->Propagate(in.RowRange(i/seq_stride_, 1), &tmp_row);
   //   tmp_out.RowRange(i, 1).CopyFromMat(tmp_row);
   //}

   //assert(Same(*out, tmp_out));
}

void EmbedMux::BackpropagateFnc(const CuMatrixBase<BaseFloat> &in, const CuMatrixBase<BaseFloat> &out,
      const CuMatrixBase<BaseFloat> &out_diff, CuMatrixBase<BaseFloat> *in_diff){

   for(int i = 0; i < cnt_.size(); ++i){
      if(cnt_[i] != 0){
         out_diff_buff_[i].Resize(cnt_[i], output_dim_, kSetZero);
      }
   }

   dist_back(out_diff, seq_device_.Data(), seq_stride_,
         id_device_.Data(), getVecCuMatrixPt(out_diff_buff_), getVecCuMatrixStride(out_diff_buff_));

   for(int i = 0; i < comps_.size(); ++i)
      if(cnt_[i] != 0){
         CuMatrix<BaseFloat> *ptr = in_diff == NULL ? NULL : &in_diff_buff_[i];
         comps_[i]->Backpropagate(in_buff_[i], out_buff_[i],
               out_diff_buff_[i], ptr);
      }

   if(in_diff != NULL){
      comb_back(getVecCuMatrixPt(in_diff_buff_), getVecCuMatrixStride(in_diff_buff_), seq_device_.Data(), seq_stride_, id_device_.Data(), *in_diff);
   // check consistence...
   //CuMatrix<BaseFloat> tmp_sum(1, output_dim_, kSetZero);
   //CuMatrix<BaseFloat> tmp_sum_diff(1, input_dim_, kSetZero);

   //for(int i = 0; i < seq_stride_; ++i)
   //   tmp_sum.AddMat(1.0, out_diff.RowRange(i, 1));
   //comps_[seq_[0]]->Backpropagate(in.RowRange(0, 1), out.RowRange(0, 1),
   //      tmp_sum, &tmp_sum_diff);

   //assert(Same(tmp_sum_diff, in_diff->RowRange(0, 1)));
   
   //CuMatrix<BaseFloat> tmp_in_diff(in.NumRows(), input_dim_, kSetZero);
   //CuMatrix<BaseFloat> tmp_row(1, input_dim_, kUndefined);
   //for(int i = 0; i < seq_.size(); ++i){
   //   comps_[seq_[i]]->Backpropagate(in.RowRange(i/seq_stride_, 1), out.RowRange(i, 1),
   //         out_diff.RowRange(i, 1), &tmp_row);
   //   tmp_in_diff.RowRange(i/seq_stride_, 1).AddMat(1.0, tmp_row);
   //}
   //assert(Same(*in_diff, tmp_in_diff));
   }
}

void EmbedMux::InitData(istream &is){
   Embed::InitData(is);
   
   string conf_line;
   assert(getline(is, conf_line));

   for(int i = 0; i < label_num_; ++i){
      comps_[i] = Component::Init(conf_line);

      // check dimensions
      assert(comps_[i]->InputDim() == input_dim_ &&
            comps_[i]->OutputDim() == output_dim_);
   }
}

void EmbedMux::ReadData(istream &is, bool binary){
   Embed::ReadData(is, binary);
   
   for(int i = 0; i < label_num_; ++i){
      comps_[i] = Component::Read(is, binary);

      assert(comps_[i]->InputDim() == input_dim_ && 
            comps_[i]->OutputDim() == output_dim_);
   }
}

void EmbedMux::WriteData(ostream &os, bool binary) const{
   Embed::WriteData(os, binary);

   for(int i = 0; i < comps_.size(); ++i)
      comps_[i]->Write(os, binary);
}


int32 EmbedMux::NumParams() const {
   int32 sum = 0;
   for(int i = 0; i < comps_.size(); ++i)
      if(comps_[i]->IsUpdatable()){
         UpdatableComponent *uc = dynamic_cast<UpdatableComponent*>(comps_[i]);
         sum += uc -> NumParams();
      }
   return sum;
}

void EmbedMux::GetParams(Vector<BaseFloat> *params) const {
   params->Resize(NumParams());
   int pos = 0;
   for(int i = 0; i < comps_.size(); ++i){
      if(comps_[i]->IsUpdatable()){
         UpdatableComponent &c = dynamic_cast<UpdatableComponent&>(*comps_[i]);
         Vector<BaseFloat> c_params;
         c.GetParams(&c_params);
         params->Range(pos, c_params.Dim()).CopyFromVec(c_params);
         pos += c_params.Dim();
      }
   }

   assert(pos == NumParams());
}

void EmbedMux::Update(const CuMatrixBase<BaseFloat> &input, const CuMatrixBase<BaseFloat> &diff){
   for(int i = 0; i < comps_.size(); ++i)
      if(comps_[i]->IsUpdatable() && cnt_[i] != 0){
         UpdatableComponent *uc = dynamic_cast<UpdatableComponent*>(comps_[i]);
         uc->Update(in_buff_[i], out_diff_buff_[i]);
      }
}

void EmbedMux::SetLabelSeqsFnc(const vector<int32> &seq, int seq_stride){
   id_.resize(seq.size());

   // clear cnt_
   for(int i = 0; i < cnt_.size(); ++i)
      cnt_[i] = 0;

   vector<int> mask(comps_.size(), 0);
   int flag = 0;

   for(int i = 0; i < seq.size(); i += seq_stride){
      flag++;

      for(int j = 0; j < seq_stride; ++j){
         mask[seq[i + j]] = flag;
         id_[i + j] = cnt_[seq[i + j]];
      }

      for(int j = 0; j < mask.size(); ++j)
         if(mask[j] == flag) cnt_[j]++;
   }

   // copy to device
   seq_device_ = seq_;
   id_device_  = id_;
}

void EmbedMux::SetLabelNumFnc(int label_num){
   comps_.resize(label_num);
   in_buff_.resize(label_num);
   in_diff_buff_.resize(label_num);
   out_buff_.resize(label_num);
   out_diff_buff_.resize(label_num);

   cnt_.resize(label_num);
}

void EmbedMux::SetTrainOptions(const NnetTrainOptions &opts){
   opts_ = opts;
   for(int i = 0; i < comps_.size(); ++i)
      if(comps_[i]->IsUpdatable()){
         UpdatableComponent *uc = dynamic_cast<UpdatableComponent*>(comps_[i]);
         uc->SetTrainOptions(opts_);
      }
}


BaseFloat** EmbedMux::getVecCuMatrixPt(vector< CuMatrix<BaseFloat> > &mat_arr){
   vector< BaseFloat* > arr(mat_arr.size());
   for(int i = 0; i < mat_arr.size(); ++i)
      arr[i] = getCuPointer(&mat_arr[i]);

   mat_arr_pt_device_ = arr;
   return mat_arr_pt_device_.Data();
}

int32* EmbedMux::getVecCuMatrixStride(vector< CuMatrix<BaseFloat> > &mat_arr){
   vector< int > arr(mat_arr.size());
   for(int i = 0; i < mat_arr.size(); ++i)
      arr[i] = mat_arr[i].Stride();

   mat_arr_stride_device_ = arr;
   return mat_arr_stride_device_.Data();
}
