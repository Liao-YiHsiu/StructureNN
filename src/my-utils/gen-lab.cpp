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
      usage.append("transform input feature key and timit path to generate label sequence\n")
         .append("Usage: ").append(argv[0]).append(" <timit path> <phone map> <phone map ids> <rspecifier> <wspecifier>\n")
         .append("e.g.: ").append(argv[0]).append(" /corpus/timit s5/conf/phones.60-48-39.map ../timit/data/lang/phones.txt scp:feats.scp ark,t:-\n");

      ParseOptions po(usage.c_str());
      po.Read(argc, argv);

      if( po.NumArgs() != 5 ){
         po.PrintUsage();
         exit(1);
      }

      string timit_path = po.GetArg(1);
      string phone_path = po.GetArg(2);
      string phone_map  = po.GetArg(3);

      map<string, int> phMap;
      readPhMap(phone_path, phone_map, phMap);

      if (ClassifyRspecifier(po.GetArg(4), NULL, NULL) != kNoRspecifier) {
         string rspecifier = po.GetArg(4);
         string wspecifier = po.GetArg(5);

         SequentialBaseFloatMatrixReader kaldi_reader(rspecifier);
         UcharVectorWriter label_writer(wspecifier);

         for (int index = 1; !kaldi_reader.Done(); kaldi_reader.Next(), index++){
            const Matrix<BaseFloat> &matrix = kaldi_reader.Value();
            vector<uchar> phIdx(matrix.NumRows());
            getPhone(kaldi_reader.Key(), timit_path, phMap, phIdx);

            label_writer.Write(kaldi_reader.Key(), phIdx);
         }
      }
   }catch(const exception &e){
      cerr << e.what() << endl;
      return -1;
   }

   return 0;
}
