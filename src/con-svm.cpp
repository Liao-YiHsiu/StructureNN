#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "matrix/kaldi-matrix.h"
#include "util.h"

using namespace std;
using namespace kaldi;

void gen(ofstream &fout, int index,const Matrix<BaseFloat> &matrix,const vector<int> &phIdx);
void getLabel(const string &path,  map<string, vector<int> > &labelMap);

int main(int argc, char* argv[]){
   // parse arguement.
   try{
      string usage;
      usage.append("Use feature and label sequence to generate structure learning feature\n")
         .append("Usage: ").append(argv[0]).append(" <label-rspecifier> <feat-rspecifier> <outFile>\n")
         .append("e.g.: ").append(argv[0]).append(" scp:label.scp scp:feats.scp out\n");

      ParseOptions po(usage.c_str());
      po.Read(argc, argv);

      if( po.NumArgs() != 3 ){
         po.PrintUsage();
         exit(1);
      }

      string label_rspecifier = po.GetArg(1);
      string feats_rspecifier = po.GetArg(2);

      if (ClassifyRspecifier(label_rspecifier, NULL, NULL) != kNoRspecifier &&
            ClassifyRspecifier(feats_rspecifier, NULL, NULL) != kNoRspecifier ) {

         ofstream fout(po.GetArg(3).c_str());

         SequentialBaseFloatMatrixReader feats_reader(feats_rspecifier);
         RandomAccessInt32VectorReader   label_reader(label_rspecifier);

         for (int index = 1; !feats_reader.Done(); feats_reader.Next(), index++){
            const Matrix<BaseFloat> &matrix = feats_reader.Value();

            assert( label_reader.HasKey(feats_reader.Key()) );

            gen(fout, index, matrix, label_reader.Value(feats_reader.Key()) );
         }
      }
   }catch(const exception &e){
      cerr << e.what() << endl;
      return -1;
   }

   return 0;
}

void gen(ofstream &fout, int index, const Matrix<BaseFloat> &matrix, const vector<int32> &phIdx){
   int F = matrix.NumCols(), T = matrix.NumRows();
   assert(T == phIdx.size());

   for(int t = 0; t < T; ++t){
      if(phIdx[t] < 0) continue;

      const SubVector<BaseFloat>& vec = matrix.Row(t);
      
      fout << phIdx[t] << " qid:" << index << " ";
      
      for(int i = 0; i < F; ++i)
         fout << i + 1 << ":" << vec(i) << " ";

      fout << endl;
   }
   
}
