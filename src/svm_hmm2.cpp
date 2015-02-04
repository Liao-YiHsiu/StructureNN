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
         .append("Usage: ").append(argv[0]).append(" <answer> <rspecifier> <outFile>\n")
         .append("e.g.: ").append(argv[0]).append(" answer scp:feats.scp out\n");

      ParseOptions po(usage.c_str());
      po.Read(argc, argv);

      if( po.NumArgs() != 3 ){
         po.PrintUsage();
         exit(1);
      }

      map<string, vector<int> > labelMap;
      string label_path = po.GetArg(1);
      getLabel(label_path, labelMap);

      if (ClassifyRspecifier(po.GetArg(2), NULL, NULL) != kNoRspecifier) {
         string rspecifier = po.GetArg(2);
         ofstream fout(po.GetArg(3).c_str());

         SequentialBaseFloatMatrixReader kaldi_reader(rspecifier);

         for (int index = 1; !kaldi_reader.Done(); kaldi_reader.Next(), index++){
            const Matrix<BaseFloat> &matrix = kaldi_reader.Value();

            assert(labelMap.find(kaldi_reader.Key()) != labelMap.end());
            gen(fout, index, matrix, labelMap[kaldi_reader.Key()]);
         }
      }
   }catch(const exception &e){
      cerr << e.what() << endl;
      return -1;
   }

   return 0;
}

void getLabel(const string &path,  map<string, vector<int> > &labelMap){
   ifstream fin(path.c_str());
   string line;
   while(getline(fin, line)){
      stringstream ss(line);
      string name;
      vector<int> phns;
      int ph;

      ss >> name;
      while(ss >> ph)
         phns.push_back(ph);

      labelMap[name] = phns;
   }
}

void gen(ofstream &fout, int index, const Matrix<BaseFloat> &matrix, const vector<int> &phIdx){
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
