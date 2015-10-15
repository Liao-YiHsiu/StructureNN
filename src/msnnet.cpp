#include "msnnet.h"

MSNnet::MSNnet(const MSNnet& other){
   nnet1_ = other.nnet1_;
   nnet2_ = other.nnet2_;
   mux_   = other.mux_->Copy();
   Check();
}

MSNnet& MSNnet::operator = (const MSNnet& other){
   Destroy();
   nnet1_ = other.nnet1_;
   nnet2_ = other.nnet2_;
   mux_   = other.mux_->Copy();
   Check();

   return *this;
}

MSNnet::~MSNnet(){
   Destroy();
}

void MSNnet::Propagate(const CuMatrixBase<BaseFloat> &in, CuMatrix<BaseFloat> *out){
   assert( out != NULL );
   assert( in.NumRows()*labels_stride_ == labels_.size() );

   nnet1_.Propagate(in, &propagate_nnet1_out_buf_);
   mux_ ->Propagate(propagate_nnet1_out_buf_, &propagate_mux_out_buf_);
   nnet2_.Propagate(propagate_mux_out_buf_,   &propagate_nnet2_out_buf_);

   int outRows = streamN_ * labels_stride_;
   out->Resize(outRows, nnet2_.OutputDim(), kSetZero);

   // sum over all output
   int T = in.NumRows()/streamN_;
   for(int i = 0; i < T; ++i)
      out->AddMat(1.0, propagate_nnet2_out_buf_.RowRange(i * outRows, outRows));
}

void MSNnet::Backpropagate(const CuMatrixBase<BaseFloat> &out_diff, CuMatrix<BaseFloat> *in_diff){
   int outRows = streamN_ * labels_stride_;
   assert(out_diff.NumRows() == outRows);

   int T = labels_.size() / outRows;
   backpropagate_out_buf_.Resize(T * outRows, nnet2_.OutputDim(), kUndefined);
   for(int i = 0; i < T; ++i){
      backpropagate_out_buf_.RowRange(i * outRows, outRows).CopyFromMat(out_diff);
   }

   nnet2_.Backpropagate(backpropagate_out_buf_, &backpropagate_nnet2_in_buf_);

   mux_ ->Backpropagate(propagate_nnet1_out_buf_, propagate_mux_out_buf_, 
         backpropagate_nnet2_in_buf_, &backpropagate_mux_in_buf_);
   mux_ ->Update(propagate_nnet1_out_buf_, backpropagate_nnet2_in_buf_);

   nnet1_.Backpropagate(backpropagate_mux_in_buf_, in_diff);
}

void MSNnet::Feedforward(const CuMatrixBase<BaseFloat> &in, CuMatrix<BaseFloat> *out){
   assert( out != NULL );
   assert( in.NumRows()*labels_stride_ == labels_.size() );

   nnet1_.Feedforward(in, &propagate_nnet1_out_buf_);
   mux_ ->Propagate(propagate_nnet1_out_buf_, &propagate_mux_out_buf_);
   nnet2_.Feedforward(propagate_mux_out_buf_,   &propagate_nnet2_out_buf_);

   int outRows = streamN_ * labels_stride_;
   out->Resize(outRows, nnet2_.OutputDim(), kSetZero);

   // sum over all output
   int T = in.NumRows()/streamN_;
   for(int i = 0; i < T; ++i)
      out->AddMat(1.0, propagate_nnet2_out_buf_.RowRange(i * outRows, outRows));
}

void MSNnet::SetNnet(const Nnet &nnet1, const Nnet &nnet2, const Mux &mux){
   nnet1_ = nnet1;
   nnet2_ = nnet2;
   mux_   = mux.Copy();

   Check();
}


int32 MSNnet::NumParams() const{
   return nnet1_.NumParams() + mux_->NumParams() + nnet2_.NumParams();
}

void MSNnet::SetDropoutRetention(BaseFloat r){
   nnet1_.SetDropoutRetention(r);
   nnet2_.SetDropoutRetention(r);
}

void MSNnet::SetLabelSeqs(const vector<int32> &labels, int labels_stride){
   assert( labels.size() % labels_stride == 0 );
   // check validation
#pragma omp parallel for
   for(int i = 0; i < labels.size(); ++i)
      assert(labels[i] < mux_->NumComponents());
   labels_ = labels;
   labels_stride_ = labels_stride;

   mux_->setSeqs(labels, labels_stride);
}

void MSNnet::ResetLstmStreams(const vector<int32> &stream_reset_flag){
   streamN_ = stream_reset_flag.size();
   assert( labels_.size() % (labels_stride_ * streamN_) == 0);

   nnet1_.ResetLstmStreams(stream_reset_flag);
   nnet2_.ResetLstmStreams(expandVec(stream_reset_flag, labels_stride_));
}

void MSNnet::SetSeqLengths(const vector<int32> &sequence_lengths){
   streamN_ = sequence_lengths.size();
   assert( labels_.size() % (labels_stride_ * streamN_) == 0);

   nnet1_.SetSeqLengths(sequence_lengths);
   nnet2_.SetSeqLengths(expandVec(sequence_lengths, labels_stride_));
}

void MSNnet::Init(const Nnet& nnet1, const Nnet& nnet2, const string &config_file){
   Input in(config_file);
   istream &is = in.Stream();

   mux_   = Mux::Init(is);
   nnet1_ = nnet1;
   nnet2_ = nnet2;

   Check();
}

void MSNnet::Read(const string &file){
   bool binary;
   Input in(file, &binary);
   Read(in.Stream(), binary);
   in.Close();
}

void MSNnet::Read(istream &is, bool binary){
   nnet1_.Read(is, binary);
   nnet2_.Read(is, binary);
   mux_ = Mux::Read(is, binary);

   Check();
}

void MSNnet::Write(const string &file, bool binary) const{
   Output out(file, binary, true);
   Write(out.Stream(), binary);
   out.Close();
}

void MSNnet::Write(ostream &os, bool binary) const{
   Check();
   nnet1_.Write(os, binary);
   nnet2_.Write(os, binary);
   mux_ ->Write(os, binary);
}

string MSNnet::Info() const{
   ostringstream ostr;
   ostr << nnet1_.Info() << endl;
   
   ostr << "Mux component : input-dim " << mux_->InputDim() << 
      ", output-dim " << mux_->OutputDim() << endl;
   ostr << mux_->Info() << endl;
   ostr << nnet2_.Info() << endl;
   return ostr.str();
}

string MSNnet::InfoGradient() const{
   return nnet1_.InfoGradient() + nnet2_.InfoGradient();
}

string MSNnet::InfoPropagate() const{
   return nnet1_.InfoPropagate() + nnet2_.InfoPropagate();
}

string MSNnet::InfoBackPropagate() const{
   return nnet1_.InfoBackPropagate() + nnet2_.InfoBackPropagate();
}

void MSNnet::Check() const{
   assert(nnet1_.OutputDim() == mux_->InputDim());
   assert(mux_->OutputDim() == nnet2_.InputDim());
}

void MSNnet::Destroy(){
   nnet1_.Destroy();
   nnet2_.Destroy();
   if(mux_) delete mux_;
   mux_ = NULL;
}

void MSNnet::SetTrainOptions(const NnetTrainOptions& opts){
   nnet1_.SetTrainOptions(opts);
   mux_ ->SetTrainOptions(opts);
   nnet2_.SetTrainOptions(opts);
}


vector<int32> MSNnet::expandVec(const vector<int32> &src, int mul){
   vector<int32> dest(src.size() * mul);
   for(int i = 0; i < src.size(); ++i)
      for(int j = 0; j < mul; ++j)
         dest[i * mul + j] = src[i];

   return dest;
}
