#include "score-path/score-path.h"

template<class V, class T>
void ValueVectorPair<V, T>::Write(ostream &os, bool binary) const{
   try{
      int vecsz = static_cast<int>(val.size());
      KALDI_ASSERT((size_t)vecsz == val.size());

      if(binary){
         os.write(reinterpret_cast<const char *>(&vecsz), sizeof(vecsz));
      }else{
         os << vecsz << " " ;
      }

      for(int i = 0; i < val.size(); ++i){
         WriteBasicType(os, binary, val[i].first);
         WriteIntegerVector(os, binary, val[i].second);
      }
   } catch(const std::exception &e) {
      std::cerr << e.what();
      exit(-1);
   }
}

template<class V, class T>
void ValueVectorPair<V, T>::Read(istream &is, bool binary){
   int vecsz;
   if(binary){
      is.read(reinterpret_cast<char *>(&vecsz), sizeof(vecsz));
   }else{
      is >> vecsz >> std::ws;
   }

   val.resize(vecsz);
   for(int i = 0; i < vecsz; ++i){
      ReadBasicType(is, binary, &val[i].first);
      ReadIntegerVector(is, binary, &val[i].second);
   }
}

// Instantiate class
template class ValueVectorPair<BaseFloat, uchar>;
