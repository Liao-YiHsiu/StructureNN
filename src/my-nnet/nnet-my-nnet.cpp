#include "nnet-my-nnet.h"
#include "nnet/nnet-component.h"
#include "nnet/nnet-parallel-component.h"
#include "nnet/nnet-activation.h"
#include "nnet/nnet-affine-transform.h"
#include "nnet/nnet-various.h"
#include "nnet/nnet-lstm-projected-streams.h"
#include "nnet/nnet-blstm-projected-streams.h"
#include "nnet-embed-component.h"
#include "nnet-blend-component.h"

#define SUBMATRIX(mat_arr, index) mat_arr[index].RowRange(0, rows_num_[index])

MyNnet::MyNnet(const MyNnet& other){
   for(int i = 0; i < other.NumComponents(); ++i){
      components_.push_back(other.GetComponent(i).Copy());
   }

   propagate_buf_.resize(NumComponents() + 1);
   backpropagate_buf_.resize(NumComponents() + 1);

   SetTrainOptions(other.opts_);

   streamN_ = other.streamN_;
   labels_stride_ = other.labels_stride_;
   embedIdx_ = other.embedIdx_;
   blendIdx_ = other.blendIdx_;

   Check();
}

MyNnet& MyNnet::operator= (const MyNnet &other){
   Destroy();
   for(int i = 0; i < other.NumComponents(); ++i){
      components_.push_back(other.GetComponent(i).Copy());
   }

   propagate_buf_.resize(NumComponents() + 1);
   backpropagate_buf_.resize(NumComponents() + 1);

   SetTrainOptions(other.opts_);
   streamN_ = other.streamN_;
   labels_stride_ = other.labels_stride_;
   embedIdx_ = other.embedIdx_;
   blendIdx_ = other.blendIdx_;

   Check();

   return *this;
}

void MyNnet::Propagate(const CuMatrixBase<BaseFloat> &in, CuMatrix<BaseFloat> *out){
   assert( out != NULL );

   if( NumComponents() == 0 ){
      (*out) = in;
      return;
   }

   assert( propagate_buf_.size() == NumComponents() + 1);

   SetBuff(in.NumRows(), labels_stride_, streamN_);

   SUBMATRIX(propagate_buf_, 0).CopyFromMat(in);

   for(int i = 0; i < components_.size(); ++i){
      //if(components_[i]->GetType() == Component::kLstmProjectedStreams || 
      //      components_[i]->GetType() == Component::kBLstmProjectedStreams){
      //   components_[i]->Propagate(propagate_buf_[i], &propagate_buf_[i+1]);
      //}else if(components_[i]->GetType() != Component::kUnknown){
      if(components_[i]->GetType() != Component::kUnknown){
         ((ComponentBuff*)(void*)components_[i])->
            Propagate(SUBMATRIX(propagate_buf_, i), &propagate_buf_[i+1]);
      }else{
         MyComponent *mycomp = dynamic_cast<MyComponent*>(components_[i]);
         mycomp->Propagate(SUBMATRIX(propagate_buf_, i), &propagate_buf_[i+1]);
      }
   }

   (*out) = SUBMATRIX(propagate_buf_, components_.size());
}

void MyNnet::Backpropagate(const CuMatrixBase<BaseFloat> &out_diff, CuMatrix<BaseFloat> *in_diff){
   if( NumComponents() == 0){
      (*in_diff) = out_diff;
      return;
   }

   assert(propagate_buf_.size() == NumComponents() + 1);
   assert(backpropagate_buf_.size() == NumComponents() + 1);

   SUBMATRIX(backpropagate_buf_, NumComponents()).CopyFromMat(out_diff);

   for(int i = NumComponents()-1; i >= 0; --i){
      //if(components_[i]->GetType() == Component::kLstmProjectedStreams || 
      //      components_[i]->GetType() == Component::kBLstmProjectedStreams){
      //   components_[i]->Backpropagate(propagate_buf_[i], propagate_buf_[i+1],
      //         backpropagate_buf_[i+1], &backpropagate_buf_[i]);
      //}else if(components_[i]->GetType() != Component::kUnknown){
      if(components_[i]->GetType() != Component::kUnknown){
         ((ComponentBuff*)(void*)components_[i])->
         Backpropagate(SUBMATRIX(propagate_buf_, i), SUBMATRIX(propagate_buf_, i+1), 
               SUBMATRIX(backpropagate_buf_, i+1), &backpropagate_buf_[i]);
      }else{
         MyComponent *mycomp = dynamic_cast<MyComponent*>(components_[i]);
         mycomp->Backpropagate(SUBMATRIX(propagate_buf_, i), SUBMATRIX(propagate_buf_, i+1),
               SUBMATRIX(backpropagate_buf_, i+1), &backpropagate_buf_[i]);
      }

      if(components_[i]->IsUpdatable()){
         UpdatableComponent *uc = dynamic_cast<UpdatableComponent*>(components_[i]);
         uc->Update(SUBMATRIX(propagate_buf_, i), SUBMATRIX(backpropagate_buf_, i+1));
      }
   }

   if(NULL != in_diff) 
      (*in_diff) = SUBMATRIX(backpropagate_buf_, 0);
}

int32 MyNnet::OutputDim() const {
   assert(!components_.empty());
   return components_.back()->OutputDim();
}

int32 MyNnet::InputDim() const{
   assert(!components_.empty());
   return components_.front()->InputDim();
}

const Component& MyNnet::GetComponent(int32 c) const{
   assert( c < components_.size() );
   return *(components_[c]);
}

Component& MyNnet::GetComponent(int32 c){
   assert( c < components_.size() );
   return *(components_[c]);
}

void MyNnet::SetComponent(int32 c, Component *comp){
   assert( c < components_.size() );
   delete components_[c];
   components_[c] = comp;
   Check();
}

void MyNnet::AppendComponent(Component *comp){
   components_.push_back(comp);

   propagate_buf_.resize(NumComponents() + 1);
   backpropagate_buf_.resize(NumComponents() + 1);

   if(comp->GetType() == Component::kUnknown){
      MyComponent* mycomp = dynamic_cast<MyComponent*>(comp);
      if(mycomp->IsEmbed()){
         assert(embedIdx_ < 0);
         embedIdx_ = NumComponents() - 1;
      }else if(mycomp->IsBlend()){
         assert(blendIdx_ < 0);
         blendIdx_ = NumComponents() - 1;
      }
   }

   Check();
}

int32 MyNnet::NumParams() const{
   int sum = 0;
   for(int i = 0; i < components_.size(); ++i){
      if(components_[i]->IsUpdatable()){
         sum += dynamic_cast<UpdatableComponent*>(components_[i])->NumParams();
      }
   }
   return sum;
}

void MyNnet::GetParams(Vector<BaseFloat> *weights) const{
   weights->Resize(NumParams());
   int pos = 0;
   for(int i = 0; i < components_.size(); ++i){
      if(components_[i]->IsUpdatable()){
         UpdatableComponent &c = dynamic_cast<UpdatableComponent&>(*components_[i]);
         Vector<BaseFloat> c_params;
         c.GetParams(&c_params);
         weights->Range(pos, c_params.Dim()).CopyFromVec(c_params);
         pos += c_params.Dim();
      }
   }

   assert(pos == NumParams());
}

void MyNnet::SetDropoutRetention(BaseFloat r){
   for(int i = 0; i < components_.size(); ++i){
      if(components_[i]->GetType() == Component::kDropout){
         Dropout& comp = dynamic_cast<Dropout&>(*components_[i]);
         comp.SetDropoutRetention(r);
      }
   }
}

int32 MyNnet::GetLabelNum() const{
   if(embedIdx_ < 0) return 1;

   Embed* embed = dynamic_cast<Embed*>(components_[embedIdx_]);
   return embed->GetLabelNum();
}

void MyNnet::SetLabelSeqs(const vector<int32> &labels, int labels_stride){
   if(embedIdx_ < 0) return;
   
   Embed* embed = dynamic_cast<Embed*>(components_[embedIdx_]);
   embed->SetLabelSeqs(labels, labels_stride);

   labels_stride_ = labels_stride;
}

void MyNnet::ResetLstmStreams(const vector<int32> &stream_reset_flag){
   vector<int32> inflat_flag;

   for(int i = 0; i < components_.size(); ++i){
      if(components_[i]->GetType() == Component::kLstmProjectedStreams){
         LstmProjectedStreams &comp = dynamic_cast<LstmProjectedStreams&>(*components_[i]);
         if(i < embedIdx_ || embedIdx_ < 0){
            comp.ResetLstmStreams(stream_reset_flag);
         }else{
            if(inflat_flag.size() == 0){
               inflat_flag.resize(stream_reset_flag.size() * labels_stride_);
               for(int i = 0; i < inflat_flag.size(); ++i){
                  inflat_flag[i] = stream_reset_flag[i/labels_stride_];
               }
            }

            comp.ResetLstmStreams(inflat_flag);
         }
      }
   }
   streamN_ = stream_reset_flag.size();
}

void MyNnet::SetSeqLengths(const vector<int32> &sequence_lengths){
   vector<int32> inflat_lengths;

   for(int i = 0; i < components_.size(); ++i){
      if(components_[i]->GetType() == Component::kBLstmProjectedStreams){
         BLstmProjectedStreams &comp = dynamic_cast<BLstmProjectedStreams&>(*components_[i]);
         if(i < embedIdx_ || embedIdx_ < 0){
            comp.SetSeqLengths(sequence_lengths);
         }else{
            if(inflat_lengths.size() == 0){
               inflat_lengths.resize(sequence_lengths.size() * labels_stride_);
               for(int i = 0; i < inflat_lengths.size(); ++i){
                  inflat_lengths[i] = sequence_lengths[i/labels_stride_];
               }
            }

            comp.SetSeqLengths(inflat_lengths);
         }
      }
   }

   if(blendIdx_ >= 0){
      Blend &blend = dynamic_cast<Blend&>(*components_[blendIdx_]);
      if(inflat_lengths.size() == 0){
         inflat_lengths.resize(sequence_lengths.size() * labels_stride_);
         for(int i = 0; i < inflat_lengths.size(); ++i){
            inflat_lengths[i] = sequence_lengths[i/labels_stride_];
         }
      }

      blend.SetSeqLengths(inflat_lengths);
   }
   streamN_ = sequence_lengths.size();
}

void MyNnet::SetBuff(int max_input_rows, int labels_stride, int streamN){
   int rows = max_input_rows;

   rows_num_.resize(propagate_buf_.size());

   rows_num_[0] = rows;
   resizeBuff(&propagate_buf_[0], rows, components_[0]->InputDim());
   resizeBuff(&backpropagate_buf_[0], rows, components_[0]->InputDim());

   for(int i = 0; i < components_.size(); ++i){
      if(i == embedIdx_)
         rows = max_input_rows * labels_stride;
      else if(i == blendIdx_)
         rows = streamN * labels_stride;

      rows_num_[i+1] = rows;
      resizeBuff(&propagate_buf_[i+1], rows, components_[i]->OutputDim());
      resizeBuff(&backpropagate_buf_[i+1], rows, components_[i]->OutputDim());
   }

   for(int i = 0; i < propagate_buf_.size(); ++i){
      propagate_buf_[i].SetZero();
      backpropagate_buf_[i].SetZero();
   }
}

void MyNnet::Init(const string &file){
   Input in(file);
   istream &is = in.Stream();
   string conf_line, token;
   while(!is.eof()){
      assert(is.good());
      getline(is, conf_line);
      if(conf_line == "") continue;
      KALDI_VLOG(1) << conf_line;
      istringstream(conf_line) >> ws >> token;
      if(token == "<NnetProto>" || token == "</NnetProto>" ||
            token == "<MyNnetProto>" || token == "</MyNnetProto>")
         continue;
      AppendComponent(MyComponent::Init(conf_line + "\n"));
      is >> ws;
   }
   
   in.Close();
   Check();
}

void MyNnet::Read(const string &file){
   bool binary;
   Input in(file, &binary);
   Read(in.Stream(), binary);
   in.Close();
}

void MyNnet::Read(istream &is, bool binary){
   Component* comp;
   while(NULL != (comp = MyComponent::Read(is, binary))){
      AppendComponent(comp);
   }

   propagate_buf_.resize(NumComponents() + 1);
   backpropagate_buf_.resize(NumComponents() + 1);


   // reset learn rate
   opts_.learn_rate = 0.0;

   Check();
}

void MyNnet::Write(const string &file, bool binary) const {
   Output out(file, binary, true);
   Write(out.Stream(), binary);
   out.Close();
}

void MyNnet::Write(ostream &os, bool binary) const {
   Check();
   WriteToken(os, binary, "<MyNnet>");
   if(!binary) os << endl;

   for(int i = 0; i < components_.size(); ++i){
      if(components_[i]->GetType() == Component::kUnknown){
         MyComponent* comp = dynamic_cast<MyComponent*>(components_[i]);
         comp->Write(os, binary);
      }else{
         components_[i]->Write(os, binary);
      }
   }

   WriteToken(os, binary, "</MyNnet>");
   if(!binary) os << endl;
}

string MyNnet::Info() const {
   ostringstream ostr;
   ostr << "num-components " << NumComponents() << endl;
   ostr << "input-dim " << InputDim() << endl;
   ostr << "output-dim " << OutputDim() << endl;
   ostr << "number-of-parameters " << NumParams()/(float)1e6 
      << " millions" << endl;
   
   for(int32 i = 0; i < NumComponents(); ++i){
      ostr << "component " << i + 1 << " : "
         << MyComponent::myCompToMarker(*components_[i])
         << ", input-dim " << components_[i]->InputDim()
         << ", output-dim " << components_[i]->OutputDim()
         << ", " << components_[i]->Info() << endl;
   }
   return ostr.str();
}

string MyNnet::InfoGradient() const {
   ostringstream ostr;

   ostr << "### Gradient stats :\n";
   for(int i = 0; i < NumComponents(); ++i){
      ostr << "Component " << i+1 << " : "
         << MyComponent::myCompToMarker(*components_[i])
         << ", " << components_[i]->InfoGradient() << endl;
   }

   return ostr.str();
}

string MyNnet::InfoPropagate() const {
   ostringstream ostr;

   ostr << "### Forward propagation buffer content :\n";
   ostr << "[0] output of <Input> " << MomentStatistics(SUBMATRIX(propagate_buf_, 0)) << endl;
   for( int i = 0; i < NumComponents(); ++i){
      ostr << "["<<i<<"] output of "
         << MyComponent::myCompToMarker(*components_[i])
         << MomentStatistics(SUBMATRIX(propagate_buf_, i+1)) << endl;
      if(Component::kParallelComponent == components_[i]->GetType()){
         ostr << dynamic_cast<ParallelComponent*>(components_[i])->InfoPropagate();
      }
   }

   return ostr.str();
}

string MyNnet::InfoBackPropagate() const {
   ostringstream ostr;

   ostr << "### Backward propagation buffer content :\n";
   ostr << "[0] diff of <Input> " << MomentStatistics(SUBMATRIX(backpropagate_buf_, 0)) << endl;
   for(int i = 0; i < NumComponents(); ++i){
      ostr << "["<<i<<"] diff-output of "
         << MyComponent::myCompToMarker(*components_[i])
         << MomentStatistics(SUBMATRIX(backpropagate_buf_, i+1)) << endl;
      if(Component::kParallelComponent == components_[i]->GetType()){
         ostr << dynamic_cast<ParallelComponent*>(components_[i])->InfoBackPropagate();
      }
   }

   return ostr.str();
}

void MyNnet::Check() const {
   assert(propagate_buf_.size() == NumComponents() + 1);
   assert(backpropagate_buf_.size() == NumComponents() +1);

   // check dim
   for(int i = 0; i + 1 < components_.size(); ++i){
      assert(components_[i] != NULL);
      assert(components_[i]->OutputDim() == components_[i+1]->InputDim());
   }

   if(embedIdx_ >= 0){
      assert(dynamic_cast<MyComponent*>(components_[embedIdx_])->IsEmbed());
   }
   if(blendIdx_ >= 0){
      assert(dynamic_cast<MyComponent*>(components_[blendIdx_])->IsBlend());
      assert(embedIdx_ >= 0);
      assert(blendIdx_ == NumComponents()-1);
   }

   // only one embedding && blend
   for(int i = 0; i < components_.size(); ++i){
      if(components_[i]->GetType() != Component::kUnknown) continue;

      const MyComponent& comp = dynamic_cast<const MyComponent&>(*components_[i]);
      assert((!comp.IsEmbed()) || i == embedIdx_);
      assert((!comp.IsBlend()) || i == blendIdx_);
   }

   // check for nan/inf
   Vector<BaseFloat> weights;
   GetParams(&weights);
   BaseFloat sum = weights.Sum();
   assert(!(KALDI_ISINF(sum)||KALDI_ISNAN(sum)));
}

void MyNnet::Destroy(){
   for(int i = 0; i < NumComponents(); ++i)
      delete components_[i];

   components_.resize(0);
   propagate_buf_.resize(0);
   backpropagate_buf_.resize(0);

   streamN_ = 1;
   labels_stride_ = 1;
   embedIdx_ = -1;
   blendIdx_ = -1;
}

void MyNnet::SetTrainOptions(const NnetTrainOptions& opts){
   opts_ = opts;
   for(int i = 0; i < NumComponents(); ++i){
      if(components_[i]->IsUpdatable()){
         dynamic_cast<UpdatableComponent*>(components_[i])->SetTrainOptions(opts);
      }
   }
}

void MyNnet::forceBlend(){
   if(embedIdx_ >= 0 && blendIdx_ < 0){
      AppendComponent(new BlendSum(OutputDim(), OutputDim()));
      Check();
   }
}
