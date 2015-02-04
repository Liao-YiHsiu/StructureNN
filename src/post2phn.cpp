#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "matrix/kaldi-matrix.h"
#include "hmm/hmm-topology.h"
#include "tree/context-dep.h"
#include "tree/clusterable-classes.h"
#include "util/text-utils.h"
#include "util.h"

using namespace std;
using namespace kaldi;
void genTranMat(Matrix<BaseFloat> &tran, const ContextDependency &ctx_dep, int MaxPhn);

int main(int argc, char* argv[]){
   // parse arguement.
   try{
      string usage;
      usage.append("Transform posterior to phone probability. \n")
         .append("Usage: ").append(argv[0]).append(" <tree-in> <rspecifier> <wspecifier>\n")
         .append("e.g.: ").append(argv[0]).append(" tree scp:feats.scp ark,t:tmp.ark\n");

      ParseOptions po(usage.c_str());
      po.Read(argc, argv);

      if( po.NumArgs() != 3 ){
         po.PrintUsage();
         exit(1);
      }

      string tree_path = po.GetArg(1);
      string rspecifier = po.GetArg(2);
      string wspecifier = po.GetArg(3);


      if (ClassifyRspecifier(rspecifier, NULL, NULL) != kNoRspecifier) {

         ContextDependency ctx_dep;
         ReadKaldiObject(tree_path, &ctx_dep);
         Matrix<BaseFloat> tran;
         genTranMat(tran, ctx_dep, 48);

         SequentialBaseFloatMatrixReader kaldi_reader(rspecifier);
         BaseFloatMatrixWriter kaldi_writer(wspecifier);

         for (int index = 1; !kaldi_reader.Done(); kaldi_reader.Next(), index++){
            const Matrix<BaseFloat> &matrix = kaldi_reader.Value();
            Matrix<BaseFloat> out(matrix.NumRows(), tran.NumRows());
            out.AddMatMat(1.0, matrix, kNoTrans, tran, kTrans, 0.0);

            kaldi_writer.Write(kaldi_reader.Key(), out);
         }
      }
   }catch(const exception &e){
      cerr << e.what() << endl;
      return -1;
   }

   return 0;
}

// tran = (phones) x (posteriors) 
void genTranMat(Matrix<BaseFloat> &tran, const ContextDependency &ctx_dep, int MaxPhn){

   int32 pdfs = ctx_dep.NumPdfs();
   const EventMap &emap = ctx_dep.ToPdfMap();

   tran.Resize(MaxPhn, pdfs);

   int32 P = ctx_dep.CentralPosition();


   for(int i = 1; i <= MaxPhn; ++i){
      EventType event;
      vector<EventAnswerType> ans;

      event.push_back(
            make_pair(
               static_cast<EventKeyType>(P),
               static_cast<EventValueType>(i)));

      emap.MultiMap(event, &ans);
      SubVector<BaseFloat> vec(tran, i-1);

      for(int j = 0; j < ans.size(); ++j)
         vec(ans[j]-1) = 1;
   }
}
