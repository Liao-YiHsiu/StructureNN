#include "nnet-mux-component.h"

void Mux::Destroy(){
   for(int i = 0; i < comps_.size(); ++i)
      delete comps_[i];
   comps_.resize(0);
}

void Mux::setSeqs(const vector<int32> &seq, int seq_stride){
   assert(seq.size() % seq_stride == 0);
   seq_ = seq;
   seq_stride_ = seq_stride;

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

void Mux::Propagate(const CuMatrixBase<BaseFloat> &in, CuMatrix<BaseFloat> *out){
   assert(in.NumRows()*seq_stride_ == seq_.size() && in.NumCols() == input_dim_);

   for(int i = 0; i < cnt_.size(); ++i)
      if(in_buff_[i].NumRows() < cnt_[i]){
         in_buff_[i].Resize(cnt_[i], input_dim_, kUndefined);
         assert(in_buff_[i].Stride() == in.Stride());
      }

   dist_prop(in, seq_device_.Data(), seq_stride_, id_device_.Data(), getVecCuMatrixPt(in_buff_));

   out->Resize(seq_.size(), output_dim_, kUndefined);

   for(int i = 0; i < comps_.size(); ++i)
      if(cnt_[i] != 0){
         comps_[i]->Propagate(in_buff_[i], &out_buff_[i]);
         assert(out_buff_[i].Stride() == out->Stride());
      }

   comb_prop(getVecCuMatrixPt(out_buff_), seq_device_.Data(), seq_stride_, id_device_.Data(), *out);
}

void Mux::Backpropagate(const CuMatrixBase<BaseFloat> &in, const CuMatrixBase<BaseFloat> &out,
      const CuMatrixBase<BaseFloat> &out_diff, CuMatrix<BaseFloat> *in_diff){
   assert(in.NumRows()*seq_stride_ == seq_.size() && in.NumCols() == input_dim_);
   assert(out.NumRows() == seq_.size() && out.NumCols() == output_dim_);
   assert(out_diff.NumRows() == seq_.size() && out_diff.NumCols() == output_dim_);

   for(int i = 0; i < cnt_.size(); ++i){
      if(out_diff_buff_[i].NumRows() < cnt_[i]){
         out_diff_buff_[i].Resize(cnt_[i], output_dim_, kUndefined);
         assert(out_diff_buff_[i].Stride() == out_diff.Stride());
      }

      out_diff_buff_[i].SetZero();
   }

   dist_back(out_diff, seq_device_.Data(), seq_stride_,
         id_device_.Data(), getVecCuMatrixPt(out_diff_buff_));

   in_diff->Resize(in.NumRows(), input_dim_, kSetZero);

   for(int i = 0; i < comps_.size(); ++i)
      if(cnt_[i] != 0){
         comps_[i]->Backpropagate(in_buff_[i], out_buff_[i],
               out_diff_buff_[i], &in_diff_buff_[i]);
         assert(in_diff_buff_[i].Stride() == in_diff->Stride());
      }

   comb_back(getVecCuMatrixPt(in_diff_buff_), seq_device_.Data(), seq_stride_, id_device_.Data(), *in_diff);
}

Mux* Mux::Init(istream &is){
   string tmp_line;
   int32 num, input_dim, output_dim;

   ExpectToken(is, false, "<Mux>");
   ReadBasicType(is, false, &num);
   ExpectToken(is, false, "<InputDim>");
   ReadBasicType(is, false, &input_dim);
   ExpectToken(is, false, "<OutputDim>");
   ReadBasicType(is, false, &output_dim);
   is >> std::ws;

   Mux *m = new Mux(input_dim, output_dim, num);
   for(int i = 0; i < num; ++i){
      assert(getline(is, tmp_line));
      m->comps_[i] = Component::Init(tmp_line);

      // check dimensions
      assert(m->comps_[i]->InputDim() == input_dim &&
            m->comps_[i]->OutputDim() == output_dim);
   }

   ExpectToken(is, false, "</Mux>");
   return m;
}

Mux* Mux::Read(istream &is, bool binary){
   ExpectToken(is, binary, "<Mux>");

   int32 num, input_dim, output_dim;
   ReadBasicType(is, binary, &num);
   ReadBasicType(is, binary, &input_dim);
   ReadBasicType(is, binary, &output_dim);

   Mux *m = new Mux(input_dim, output_dim, num);
   for(int i = 0; i < num; ++i){
      m->comps_[i] = Component::Read(is, binary);

      assert(m->comps_[i]->InputDim() == input_dim && 
            m->comps_[i]->OutputDim() == output_dim);
   }
   ExpectToken(is, binary, "</Mux>");

   return m;
}

void Mux::Write(ostream &os, bool binary) const{
   WriteToken(os, binary, "<Mux>");
   int32 num = comps_.size();
   int32 input_dim = input_dim_;
   int32 output_dim = output_dim_;
   WriteBasicType(os, binary, num);
   WriteBasicType(os, binary, input_dim);
   WriteBasicType(os, binary, output_dim);

   for(int i = 0; i < comps_.size(); ++i)
      comps_[i]->Write(os, binary);
   WriteToken(os, binary, "</Mux>");
}

string Mux::Info() const{
   ostringstream os;
   for(int i = 0; i < comps_.size(); ++i)
      os << "Component " << (i+1) << ": \n" << comps_[i]->Info();

   return os.str();
}

string Mux::InfoGradient() const{
   ostringstream os;
   for(int i = 0; i < comps_.size(); ++i)
      os << "Component " << (i+1) << ": \n" << comps_[i]->InfoGradient();

   return os.str();
}

Mux* Mux::Copy() const{
   int size = comps_.size();
   Mux *m = new Mux(input_dim_, output_dim_, size);

   for(int i = 0; i < size; ++i)
      m->comps_[i] = comps_[i]->Copy();

   return m;
}

int32 Mux::NumParams() const {
   int32 sum = 0;
   for(int i = 0; i < comps_.size(); ++i)
      if(comps_[i]->IsUpdatable()){
         UpdatableComponent *uc = dynamic_cast<UpdatableComponent*>(comps_[i]);
         sum += uc -> NumParams();
      }
   return sum;
}

void Mux::Update(const CuMatrixBase<BaseFloat> &input, const CuMatrixBase<BaseFloat> &diff){
   for(int i = 0; i < comps_.size(); ++i)
      if(comps_[i]->IsUpdatable() && cnt_[i] != 0){
         UpdatableComponent *uc = dynamic_cast<UpdatableComponent*>(comps_[i]);
         uc->Update(in_buff_[i], out_diff_buff_[i]);
      }
}

void Mux::SetTrainOptions(const NnetTrainOptions &opts){
   opts_ = opts;
   for(int i = 0; i < comps_.size(); ++i)
      if(comps_[i]->IsUpdatable()){
         UpdatableComponent *uc = dynamic_cast<UpdatableComponent*>(comps_[i]);
         uc->SetTrainOptions(opts_);
      }
}


BaseFloat** Mux::getVecCuMatrixPt(vector< CuMatrix<BaseFloat> > &mat_arr){
   vector< BaseFloat* > arr(mat_arr.size());
   for(int i = 0; i < mat_arr.size(); ++i)
      arr[i] = getCuPointer(&mat_arr[i]);

   mat_arr_pt_device_ = arr;
   return mat_arr_pt_device_.Data();
}
