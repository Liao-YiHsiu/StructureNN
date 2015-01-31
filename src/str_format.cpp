#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "matrix/kaldi-matrix.h"

#define BUFSIZE 1024

using namespace std;
using namespace kaldi;

void getPhone(const string &key, const string &timit, map<string, int> &phMap, vector<int> &phIdx);
void readPhMap(const string path, map<string, int> &phMap);
void generate(Matrix<BaseFloat> &ret, const Matrix<BaseFloat> &matrix, const vector<int> &phIdx, int N);
string execute(const string &cmd);

int main(int argc, char* argv[]){
   // parse arguement.
   try{
      string usage;
      usage.append("Use DNN output sequence and label sequence to generate structure learning feature\n")
         .append("Usage: ").append(argv[0]).append(" <timit path> <phone map> <rspecifier> <wspecifier>\n")
         .append("e.g.: ").append(argv[0]).append(" /corpus/timit s5/conf/phones.60-48-39.map scp:feats.scp ark:my_feat.ark\n");

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
         string wspecifier = po.GetArg(4);

         BaseFloatMatrixWriter kaldi_writer(wspecifier);
         SequentialBaseFloatMatrixReader kaldi_reader(rspecifier);

         for (; !kaldi_reader.Done(); kaldi_reader.Next()){
            const Matrix<BaseFloat> &matrix = kaldi_reader.Value();
            Matrix<BaseFloat> ret;

            vector<int> phIdx(matrix.NumRows());
            getPhone(kaldi_reader.Key(), timit_path, phMap, phIdx);
            generate(ret, matrix, phIdx, phMap.size());

            kaldi_writer.Write(kaldi_reader.Key(), ret);
         }
      }
   }catch(const exception &e){
      cerr << e.what() << endl;
      return -1;
   }

   return 0;
}

void readPhMap(const string path, map<string, int> &phMap){

   ifstream fin(path.c_str());
   string line, tmp;
   int id = 0;

   while(getline(fin, line)){
      stringstream ss(line);
      ss >> tmp;
      phMap[tmp] = id++;
   }
}

void getPhone(const string &key, const string &timit, map<string, int> &phMap, vector<int> &phIdx){

   // get phone path
   size_t pos = key.find('_');
   string spk = key.substr(0, pos);
   string sen = key.substr(pos+1);

   char buf[BUFSIZE];
   sprintf(buf, "ls %s/*/*/%s/%s.phn", timit.c_str(), spk.c_str(), sen.c_str());
   string path = execute(buf); 

   //analyze phone
   ifstream fin(path.substr(0, path.find('\n')).c_str());
   int s, e;
   string phn;

   vector<int> ph_e;
   vector<int> ph_idx;

   while(fin >> s >> e >> phn){
      ph_e.push_back(e);
      ph_idx.push_back(phMap[phn]);
   }

   double step = ph_e[ph_e.size()-1] /(double) phIdx.size();
   int j = 0;
   for(int i = 0, size = phIdx.size(); i < size; ++i){
      double now = step/2 + i*step;
      while(now >= ph_e[j]) j++;
      phIdx[i] = ph_idx[j];
   }
}

string execute(const string &cmd){
   FILE* pipe = popen(cmd.c_str(), "r");
   assert(pipe);

   string ret;
   char buf[BUFSIZE];
   while(!feof(pipe)){
      if( fgets(buf, BUFSIZE, pipe) != NULL)
         ret.append(buf);
   }
   pclose(pipe);
   return ret;
}

void generate(Matrix<BaseFloat> &ret,const Matrix<BaseFloat> &matrix, const vector<int> &phIdx, int N){
   int F = matrix.NumCols(), T = matrix.NumRows();
   assert(T == phIdx.size());
   ret.Resize(1, F*N + N*N);
   SubVector<BaseFloat> vec(ret, 0);

   SubVector<BaseFloat> tran(vec, F*N, N*N);
   for(int t = 0; t < T; ++t){
      SubVector<BaseFloat> obs(vec, phIdx[t]*F, F);
      obs.AddVec(1.0, matrix.Row(t));
      if(t > 0){
         tran(phIdx[t-1]*N + phIdx[t]) += 1;
      }
   }
}
