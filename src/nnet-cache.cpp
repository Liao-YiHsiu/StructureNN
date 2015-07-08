#include "nnet-cache.h"

CNnet::~CNnet(){
   Destroy();
}

void CNnet::Propagate(const CuMatrixBase<BaseFloat> &in, CuMatrix<BaseFloat> *out, int index) {

   // -----------------------------------------------
   // setup buffers
   if (index >= propagate_buf_arr_.size()) {
      propagate_buf_arr_.resize(index + 1);
      backpropagate_buf_arr_.resize(index + 1);

      resizeAll();
   }

   vector<CuMatrix<BaseFloat> >& propagate_buf_ = propagate_buf_arr_[index];
   vector<CuMatrix<BaseFloat> >& backpropagate_buf_ = backpropagate_buf_arr_[index];
   vector<Component*> components_;
   GetComponents(components_);
   // -----------------------------------------------

   // kaldi original code
   KALDI_ASSERT(NULL != out);

   if (NumComponents() == 0) {
      (*out) = in; // copy 
      return; 
   }

   // we need at least L+1 input buffers
   KALDI_ASSERT((int32)propagate_buf_.size() >= NumComponents()+1);

   propagate_buf_[0].Resize(in.NumRows(), in.NumCols());
   propagate_buf_[0].CopyFromMat(in);

   for(int32 i=0; i<(int32)components_.size(); i++) {
      components_[i]->Propagate(propagate_buf_[i], &propagate_buf_[i+1]);
   }
  
   (*out) = propagate_buf_[components_.size()];
}

// computes backpropagation only, no update.
void CNnet::Backpropagate(const CuMatrixBase<BaseFloat> &out_diff, CuMatrix<BaseFloat> *in_diff, int index){
   // -----------------------------------------------
   // setup buffers
   KALDI_ASSERT(index >= propagate_buf_arr_.size());

   vector<CuMatrix<BaseFloat> >& propagate_buf_ = propagate_buf_arr_[index];
   vector<CuMatrix<BaseFloat> >& backpropagate_buf_ = backpropagate_buf_arr_[index];
   vector<Component*> components_;
   GetComponents(components_);
   // -----------------------------------------------

   //////////////////////////////////////
   // Backpropagation
   //

   // 0 layers
   if (NumComponents() == 0) { (*in_diff) = out_diff; return; }

   KALDI_ASSERT((int32)propagate_buf_.size() == NumComponents()+1);
   KALDI_ASSERT((int32)backpropagate_buf_.size() == NumComponents()+1);

   // copy out_diff to last buffer
   backpropagate_buf_[NumComponents()] = out_diff;
   // backpropagate using buffers
   for (int32 i = NumComponents()-1; i >= 0; i--) {
      components_[i]->Backpropagate(propagate_buf_[i], propagate_buf_[i+1],
            backpropagate_buf_[i+1], &backpropagate_buf_[i]);
//      if (components_[i]->IsUpdatable()) {
//         UpdatableComponent *uc = dynamic_cast<UpdatableComponent*>(components_[i]);
//         uc->Update(propagate_buf_[i], backpropagate_buf_[i+1]);
//      }
   }
   // eventually export the derivative
   if (NULL != in_diff) (*in_diff) = backpropagate_buf_[0];

   //
   // End of Backpropagation
   //////////////////////////////////////
}

void CNnet::Update(){
   // setup training options.
   // without momentant and l1, l2 norm.
   NnetTrainOptions bkup_opts = GetTrainOptions();
   NnetTrainOptions tmp_opts;
   tmp_opts.learn_rate =  bkup_opts.learn_rate;
   SetTrainOptions(tmp_opts);

   vector<Component*> components_;
   GetComponents(components_);

   for (int index = 0; index < propagate_buf_arr_.size(); ++index){
      vector<CuMatrix<BaseFloat> >& propagate_buf_ = propagate_buf_arr_[index];
      vector<CuMatrix<BaseFloat> >& backpropagate_buf_ = backpropagate_buf_arr_[index];

      for (int32 i = NumComponents()-1; i >= 0; i--) {
         if (components_[i]->IsUpdatable()) {
            UpdatableComponent *uc = dynamic_cast<UpdatableComponent*>(components_[i]);
            uc->Update(propagate_buf_[i], backpropagate_buf_[i+1]);
         }
      }
   }

   // setup training options
   SetTrainOptions(bkup_opts);
}

void CNnet::GetComponents(vector<Component*> &components){
   components.resize(NumComponents());

   for(int i = 0; i < components.size(); ++i)
      components[i] = &GetComponent(i);
}

void CNnet::Check() const {

}

void CNnet::Destroy(){
   propagate_buf_arr_.resize(0);
   backpropagate_buf_arr_.resize(0);
}

void CNnet::resizeAll(){
   KALDI_ASSERT(propagate_buf_arr_.size() == 
         backpropagate_buf_arr_.size());

   int L = NumComponents() + 1;
   for(int i = 0; i < propagate_buf_arr_.size(); ++i){
      if(propagate_buf_arr_[i].size() != L){
         propagate_buf_arr_[i].resize(L);
      }
      if(backpropagate_buf_arr_[i].size() != L){
         backpropagate_buf_arr_[i].resize(L);
      }
   }
}

