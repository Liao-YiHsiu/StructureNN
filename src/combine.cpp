#include <iostream>
#include <fstream>
#include <sstream>
#include <stdlib.h>
#include <string>
#include <vector>
#include <assert.h>

using namespace std;

void Usage(const char* progName){
   cerr << "Usage: " << progName << " <out_model> <in_model1> <in_model2> ..." << endl;
   exit(-1);
}

void readModel(const char* file, string &header, vector<double>& weight, bool add = false);
void writeModel(const char* file, const string &header, vector<double>& weight);

int main(int argc, char* argv[]){
   if(argc < 3)
      Usage(argv[0]);

   string header;
   vector<double> weight;

   readModel(argv[2], header, weight);
   for(int i = 3; i < argc; ++i)
      readModel(argv[i], header, weight, true);

   // average
   for(int i = 0; i < weight.size(); ++i)
      weight[i] = weight[i]/(argc-2);

   writeModel(argv[1], header, weight);
   return 0;
}

void readModel(const char* file, string &header, vector<double>& weight, bool add){
   ifstream fin(file);
   int featureN;

   string line;
   string tmp;
   for(int i = 0; i < 15; ++i){
      getline(fin, line);
      tmp.append(line);
      tmp.append("\n");
      if(i == 7){
         stringstream ss(line);
         ss >> featureN;
      }
   }

   if(!add){
      weight.clear();
      weight.resize(featureN, 0);
   }else{
      assert( tmp.compare(header) == 0 );
   }
   header = tmp;

   int id;
   double value;
   char del;
   
   // feature line
   getline(fin, line);
   stringstream ss(line);
   ss >> id;
   while(ss >> id >> del >> value){
      if(add){
         weight[id] += value;
      }else
         weight[id] = value;
   }

}

void writeModel(const char* file, const string &header, vector<double>& weight){
   ofstream fout(file);
   fout << header;
   fout << "1 ";
   for(int i = 0; i < weight.size(); ++i){
      if(weight[i] != 0)
         fout << i << ":" <<  weight[i] << " ";
   }
   fout << "#";
}
