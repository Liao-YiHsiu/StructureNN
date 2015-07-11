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

#include <fstream>
#define MAX_MAP 10240

using namespace std;
using namespace kaldi;
using namespace kaldi::nnet1;

int main(int argc, char *argv[]) {
  
  try {
    string usage;
    usage.append("convert two Score-path file into points (python file format).\n")
       .append("Usage: ").append(argv[0]).append(" [options] <score-path1-rspecifier> <score-path2-rspecifier>\n")
       .append("e.g.: \n")
       .append(" ").append(argv[0]).append(" ark:score-path1.ark ark:score-path2.ark \n");

    ParseOptions po(usage.c_str());

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    string score_path_rspecifier1  = po.GetArg(1);
    string score_path_rspecifier2  = po.GetArg(2);

    SequentialScorePathReader score_path_reader1(score_path_rspecifier1);  
    SequentialScorePathReader score_path_reader2(score_path_rspecifier2);  

    cout << "{ \\" << endl;
    int num_done = 0;
    for(; !score_path_reader1.Done() && !score_path_reader2.Done();
          score_path_reader1.Next(), score_path_reader2.Next()){

       assert(score_path_reader1.Key() == score_path_reader2.Key());
       const ScorePath::Table &table1 = score_path_reader1.Value().Value();
       const ScorePath::Table &table2 = score_path_reader2.Value().Value();

       // check
       assert(table1.size() == table2.size());
       for(int i = 0; i < table1.size(); ++i){

          const vector<uchar> &lab1 = table1[i].second;
          const vector<uchar> &lab2 = table2[i].second;

          // check if lab1 == lab2
          for(int j = 0; j < lab1.size(); ++j)
             assert(lab1[j] == lab2[j]);
       }

       if(num_done != 0)
          cout << ", ";
       cout << "'" << score_path_reader1.Key() << "': ( [";

       for(int i = 0; i < table1.size(); ++i){
          if( i != 0)
             cout << ", ";
          cout << table1[i].first;
       }
       cout << "], [";

       for(int i = 0; i < table2.size(); ++i){
          if( i != 0)
             cout << ", ";
          cout << table2[i].first;
       }
       cout << "] ) \\" << endl;

       num_done++;
    }
    cout << "}" << endl;

    KALDI_LOG << "Finished " << num_done;
    

    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}



