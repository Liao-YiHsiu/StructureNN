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

int main(int argc, char* argv[]){
   try{
      string usage;
      usage.append("Replace one col feature in a feature matrix\n")
         .append("usage: ").append(argv[0]).append(" <feat-rspecifier> <feat-wspecifier> <col> <value>\n")
         .append("e.g.: ").append(argv[0]).append(" scp:feat.scp ark:out.ark 0 1.5\n");

      ParseOptions po(usage.c_str());
      po.Read(argc, argv);

      if( po.NumArgs() != 4 ){
         po.PrintUsage();
         exit(1);
      }

      string feats_rspecifier = po.GetArg(1);
      string feats_wspecifier = po.GetArg(2);

      int col_index = atoi(po.GetArg(3).c_str());
      double value  = atof(po.GetArg(4).c_str());

      SequentialBaseFloatMatrixReader feats_reader(feats_rspecifier);
      BaseFloatMatrixWriter           feats_writer(feats_wspecifier);

      int num_done;
      for(num_done = 0; !feats_reader.Done(); feats_reader.Next(), num_done++){
         Matrix<BaseFloat> matrix = feats_reader.Value();

         for(int j = 0; j < matrix.NumRows(); ++j)
            matrix(j, col_index) =  value;
         
         feats_writer.Write(feats_reader.Key(), matrix);
      }

      KALDI_LOG << "Finish " << num_done;
      return 0;

   }catch(const exception &e){
      cerr << e.what() << endl;
      return -1;
   }

}
