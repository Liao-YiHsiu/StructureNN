#include "nnet/nnet-trnopts.h"
#include "nnet/nnet-nnet.h"
#include "nnet/nnet-loss.h"
#include "nnet/nnet-randomizer.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "base/timer.h"
#include "cudamatrix/cu-device.h"
#include "svm.h"
#include <sstream>

using namespace std;
using namespace kaldi;
using namespace kaldi::nnet1;

int main(int argc, char *argv[]) {
  
  try {
    string usage;
    usage.append("Combine serveral Score-path files into one.\n")
       .append("Usage: ").append(argv[0]).append(" [options] <score-path-wspecifier> <score-path1-rspecifier> [<score-path2-rspecifier> ...] \n")
       .append("e.g.: \n")
       .append(" ").append(argv[0]).append(" ark:out.ark ark:path1.ark ark:path2.ark\n");

    ParseOptions po(usage.c_str());

    bool neglect = true;
    po.Register("neglect", &neglect, "neglect duplitated path");


    po.Read(argc, argv);

    if (po.NumArgs() < 2) {
      po.PrintUsage();
      exit(1);
    }

    string score_path_wspecifier  = po.GetArg(1);
    vector<string> score_path_rspecifiers(po.NumArgs() - 1);
    for(int i = 0; i < po.NumArgs()-1; ++i)
       score_path_rspecifiers[i] = po.GetArg(i+2);


    ScorePathWriter                   score_path_writer(score_path_wspecifier);
    vector<SequentialScorePathReader> score_path_readers(score_path_rspecifiers.size());
    for(int i = 0; i < score_path_rspecifiers.size(); ++i)
       score_path_readers[i].Open(score_path_rspecifiers[i]);

    bool finish = false;
    while(!finish){
       for(int i = 0; i < score_path_readers.size(); ++i)
          if(score_path_readers[i].Done()){
             finish = true;
             break;
          }
       if(finish) break;

       // all keys should be the same.
       const string& key = score_path_readers[0].Key();
       for(int i = 1; i < score_path_readers.size(); ++i)
          assert(score_path_readers[i].Key() == key);

       ScorePath::Table table;
       for(int i = 0; i < score_path_readers.size(); ++i){
          const ScorePath::Table& tab = score_path_readers[i].Value().Value();
          for(int j = 0; j < tab.size(); ++j){
             if(neglect)
                table.push_back(tab[j]);
             else{
                // search in table for tab[j] 
                bool insert = true;
                for(int k = 0; k < table.size(); ++k){
                   if(table[k].second == tab[j].second){
                      insert = false;
                      break;
                   }
                }
                if(insert)
                   table.push_back(tab[j]);
             }
          }
       }

       score_path_writer.Write(key, ScorePath(table));
       
       for(int i = 0; i < score_path_readers.size(); ++i)
          score_path_readers[i].Next();

    }

    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}



