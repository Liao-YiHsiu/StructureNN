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

int main(int argc, char* argv[]){
   // parse arguement.
   try{
      string usage;
      usage.append("Use DNN output sequence and label sequence to generate structure learning feature\n")
         .append("Usage: ").append(argv[0]).append(" <timit path> <phone map> <rspecifier> <outFile>\n")
         .append("e.g.: ").append(argv[0]).append(" /corpus/timit s5/conf/phones.60-48-39.map scp:feats.scp out\n");

      ParseOptions po(usage.c_str());
      po.Read(argc, argv);

      if( po.NumArgs() != 4 ){
         po.PrintUsage();
         exit(1);
      }

      string timit_path = po.GetArg(1);
      string phone_path = po.GetArg(2);

      map<string, int> phMap;
      readPhMap(phone_path, phMap);

      if (ClassifyRspecifier(po.GetArg(3), NULL, NULL) != kNoRspecifier) {
         string rspecifier = po.GetArg(3);
         ofstream fout(po.GetArg(4).c_str());

         SequentialBaseFloatMatrixReader kaldi_reader(rspecifier);

         for (int index = 1; !kaldi_reader.Done(); kaldi_reader.Next(), index++){
            const Matrix<BaseFloat> &matrix = kaldi_reader.Value();

            vector<int> phIdx(matrix.NumRows());
            getPhone(kaldi_reader.Key(), timit_path, phMap, phIdx);

            gen(fout, index, matrix, phIdx);
         }
      }
   }catch(const exception &e){
      cerr << e.what() << endl;
      return -1;
   }

   return 0;
}

void gen(ofstream &fout, int index, const Matrix<BaseFloat> &matrix, const vector<int> &phIdx){
   int F = matrix.NumCols(), T = matrix.NumRows();
   assert(T == phIdx.size());

   for(int t = 0; t < T; ++t){
      const SubVector<BaseFloat>& vec = matrix.Row(t);
      
      fout << phIdx[t] + 1 << " qid:" << index << " ";
      
      for(int i = 0; i < F; ++i)
         fout << i + 1 << ":" << vec(i) << " ";

      fout << endl;
   }
   
}
