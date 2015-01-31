#ifndef __UTIL__H
#define __UTIL__H
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
string execute(const string &cmd);

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

#endif
