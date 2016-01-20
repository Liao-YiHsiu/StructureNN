#ifndef _SCORE_PATH_H_
#define _SCORE_PATH_H_

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "my-utils/type.h"

using namespace std;
using namespace kaldi;

template<class V, class T> class ValueVectorPair{
   public:
      friend class ValueVectorPair<float, int>;
      typedef vector< pair < V, vector<T> > > Table;

      ValueVectorPair(){}
      ValueVectorPair(const Table& value): val(value){}

      void Write(ostream &os, bool binary) const;
      void Read(istream &is, bool binary);

      const Table& Value() const{ return val; }
      Table& Value(){ return val; }

   private:
      Table val;
};

typedef ValueVectorPair<BaseFloat, uchar> ScorePath;

typedef SequentialTableReader<KaldiObjectHolder<ScorePath> > SequentialScorePathReader;
typedef TableWriter<KaldiObjectHolder<ScorePath> >           ScorePathWriter;

typedef SequentialTableReader<BasicVectorHolder<uchar> >     SequentialUcharVectorReader;
typedef RandomAccessTableReader<BasicVectorHolder<uchar> >   RandomAccessUcharVectorReader;
typedef TableWriter<BasicVectorHolder<uchar> >               UcharVectorWriter;

#endif
