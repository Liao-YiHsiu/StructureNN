#include "nnet-my-component.h"

#include "nnet-embed-component.h"
#include "nnet-blend-component.h"

const struct MyComponent::my_key_value MyComponent::myMarkerMap[] = {
   {MyComponent::mEmbedSimple, "<EmbedSimple>"},
   {MyComponent::mEmbedMux, "<EmbedMux>"},
   {MyComponent::mBlendSum, "<BlendSum>"}
};

const char* MyComponent::myTypeToMarker(MyComponent::MyType t){
   int32 N = sizeof(myMarkerMap)/sizeof(myMarkerMap[0]);
   for(int i = 0; i < N; ++i)
      if( myMarkerMap[i].key == t)
         return myMarkerMap[i].value;
   assert(false);
   return NULL;
}

MyComponent::MyType MyComponent::myMarkerToType(const string &s){
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
   return mUnknown;
}

const char* MyComponent::myCompToMarker(const Component &comp){
   if(comp.GetType() == Component::kUnknown){
      const MyComponent& mycomp = dynamic_cast<const MyComponent&>(comp);
      return MyComponent::myTypeToMarker(mycomp.myGetType());
   }else{
      return Component::TypeToMarker(comp.GetType());
   }
}

Component* MyComponent::Init(const string &conf_line){
   istringstream is(conf_line);
   string component_type_string;
   int32 input_dim, output_dim;

   ReadToken(is, false, &component_type_string);
   MyComponent::MyType mtype = myMarkerToType(component_type_string);

   if(mtype == mUnknown) return Component::Init(conf_line);
   
   ExpectToken(is, false, "<InputDim>");
   ReadBasicType(is, false, &input_dim);
   ExpectToken(is, false, "<OutputDim>");
   ReadBasicType(is, false, &output_dim);

   MyComponent* ans = MyComponent::NewMyComponentOfType(mtype, input_dim, output_dim);
   ans->InitData(is);

   return ans;
}

Component* MyComponent::Read(istream &is, bool binary){
   int32 dim_out, dim_in;
   string token;
   
   int first_char = Peek(is, binary);
   if(first_char == EOF)return NULL;

   streampos pos = is.tellg();
   ReadToken(is, binary, &token);

   if(token == "<Nnet>" || token == "<MyNnet>"){
      pos = is.tellg();
      ReadToken(is, binary, &token);
   }

   if(token == "</Nnet>" || token == "</MyNnet>"){
      return NULL;
   }

   MyComponent::MyType mtype = myMarkerToType(token);
   if(mtype == MyComponent::mUnknown){
      is.seekg(pos);
      return Component::Read(is, binary);
   }

   ReadBasicType(is, binary, &dim_out);
   ReadBasicType(is, binary, &dim_in);
   MyComponent *ans = NewMyComponentOfType(mtype, dim_in, dim_out);
   ans->ReadData(is, binary);
   return ans;
}

void MyComponent::Write(ostream &os, bool binary) const{
   WriteToken(os, binary, myTypeToMarker(myGetType()));
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
      case mEmbedMux:
         ans = new EmbedMux(input_dim, output_dim);
         break;
      case mBlendSum:
         ans = new BlendSum(input_dim, output_dim);
         break;
      default:
         assert(false);
   }
   return ans;
}
