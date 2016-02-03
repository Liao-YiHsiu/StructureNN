#include "my-nnet/nnet-my-component.h"

#include "my-nnet/nnet-embed-component.h"
#include "my-nnet/nnet-blend-component.h"
#include "my-nnet/nnet-activ-component.h"
#include "my-nnet/nnet-my-lstm-component.h"
#include "my-nnet/nnet-my-affine-component.h"

const struct MyComponent::key_value MyComponent::MarkerMap[] = {
   {MyComponent::mEmbedSimple, "<EmbedSimple>"},
   {MyComponent::mBlendSum, "<BlendSum>"},
   {MyComponent::mReLU, "<mReLU>"},
   {MyComponent::mSoftmax, "<mSoftmax>"},
   {MyComponent::mSigmoid, "<mSigmoid>"},
   {MyComponent::mTanh, "<mTanh>"},
   {MyComponent::mDropout, "<mDropout>"},
   {MyComponent::mLSTM, "<mLSTM>"},
   {MyComponent::mAffine, "<mAffine>"}
};

const char* MyComponent::TypeToMarker(MyComponent::MyType t){
   int32 N = sizeof(MarkerMap)/sizeof(MarkerMap[0]);
   for(int i = 0; i < N; ++i)
      if( MarkerMap[i].key == t)
         return MarkerMap[i].value;
   assert(false);
   return NULL;
}

MyComponent::MyType MyComponent::MarkerToType(const string &s){
   string s_lower(s);
   transform(s_lower.begin(), s_lower.end(), s_lower.begin(), ::tolower);
   int32 N = sizeof(MarkerMap)/sizeof(MarkerMap[0]);
   for(int i = 0; i < N; ++i){
      string m(MarkerMap[i].value);
      string m_lower(m);
      transform(m_lower.begin(), m_lower.end(), m_lower.begin(), ::tolower);
      if( m_lower == s_lower )
         return MarkerMap[i].key;
   }
   return mUnknown;
}

inline void MyComponent::Propagate(const CuMatrixBase<BaseFloat> &in,
      MyCuMatrix<BaseFloat> *out){
   assert( input_dim_ == in.NumCols() );

   out->Resize(in.NumRows(), output_dim_, kSetZero);
   PropagateFnc(in, out);
}

void MyComponent::Backpropagate(const CuMatrixBase<BaseFloat> &in,
      const CuMatrixBase<BaseFloat> &out,
      const CuMatrixBase<BaseFloat> &out_diff,
      MyCuMatrix<BaseFloat> *in_diff){

   assert(output_dim_ == out_diff.NumCols());
   assert((in.NumRows() == out.NumRows()) &&
         (in.NumRows() == out_diff.NumRows()) &&
         (out.NumCols() == out_diff.NumCols()));

   if(in_diff != NULL){
      in_diff->Resize(out_diff.NumRows(), input_dim_);
      BackpropagateFnc(in, out, out_diff, in_diff);
   }else{
      BackpropagateFnc(in, out, out_diff, NULL);
   }

   if(IsUpdatable()) Update(in, out_diff);
}

MyComponent* MyComponent::Init(const string &conf_line){
   istringstream is(conf_line);
   string component_type_string;
   int32 input_dim, output_dim;

   ReadToken(is, false, &component_type_string);
   MyComponent::MyType mtype = MarkerToType(component_type_string);
   assert(mtype != mUnknown);
   
   ExpectToken(is, false, "<InputDim>");
   ReadBasicType(is, false, &input_dim);
   ExpectToken(is, false, "<OutputDim>");
   ReadBasicType(is, false, &output_dim);

   MyComponent* ans = MyComponent::NewMyComponentOfType(mtype, input_dim, output_dim);
   ans->InitData(is);

   return ans;
}

MyComponent* MyComponent::Read(istream &is, bool binary){
   int32 dim_out, dim_in;
   string token;
   
   int first_char = Peek(is, binary);
   if(first_char == EOF)return NULL;

   ReadToken(is, binary, &token);

   if(token == "<Nnet>" || token == "<MyNnet>"){
      ReadToken(is, binary, &token);
   }

   if(token == "</Nnet>" || token == "</MyNnet>"){
      return NULL;
   }

   MyComponent::MyType mtype = MarkerToType(token);

   ReadBasicType(is, binary, &dim_out);
   ReadBasicType(is, binary, &dim_in);
   MyComponent *ans = NewMyComponentOfType(mtype, dim_in, dim_out);
   ans->ReadData(is, binary);
   return ans;
}

void MyComponent::Write(ostream &os, bool binary) const{
   WriteToken(os, binary, TypeToMarker(GetType()));
   WriteBasicType(os, binary, OutputDim());
   WriteBasicType(os, binary, InputDim());
   if(!binary) os << "\n";
   this->WriteData(os, binary);
}

MyComponent* MyComponent::NewMyComponentOfType(MyComponent::MyType type, int32 input_dim, int32 output_dim){
   MyComponent *ans = NULL;
   switch(type){
      case mEmbedSimple:
         ans = new EmbedSimple(input_dim, output_dim);
         break;
      case mBlendSum:
         ans = new BlendSum(input_dim, output_dim);
         break;
      case mReLU:
         ans = new myReLU(input_dim, output_dim);
         break;
      case mSoftmax:
         ans = new mySoftmax(input_dim, output_dim);
         break;
      case mSigmoid:
         ans = new mySigmoid(input_dim, output_dim);
         break;
      case mTanh:
         ans = new myTanh(input_dim, output_dim);
         break;
      case mDropout:
         ans = new myDropout(input_dim, output_dim);
         break;
      case mLSTM:
         ans = new myLSTM(input_dim, output_dim);
         break;
      case mAffine:
         ans = new myAffine(input_dim, output_dim);
         break;
      default:
         assert(false);
   }
   return ans;
}

