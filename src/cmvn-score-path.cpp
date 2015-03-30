#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include <math.h>
#include <cfloat>
#include "svm.h"

using namespace std;
using namespace kaldi;

int main(int argc, char *argv[]) {
  try {

    string usage;
    usage.append("Do Ceptral Mean Normal Deviation on core-path\n")
       .append("Usage: ").append(argv[0]).append(" [options] <score-path-rspecifier> <score-path-wspecifier> \n")
       .append(" e.g.: ").append(argv[0]).append(" ark:path_in.ark ark:path_out.ark \n");

    ParseOptions po(usage.c_str());
    po.Read(argc, argv);

    if (po.NumArgs() != 2){
      po.PrintUsage();
      exit(1);
    }

    string score_path_rspecifier = po.GetArg(1);
    string score_path_wspecifier = po.GetArg(2);

    SequentialScorePathReader score_path_reader(score_path_rspecifier);
    ScorePathWriter           score_path_writer(score_path_wspecifier);

    int32 n_done = 0;
    for(;!score_path_reader.Done(); score_path_reader.Next()){
       ScorePath     score = score_path_reader.Value();
       const string& key   = score_path_reader.Key(); 

       ScorePath::Table &table = score.Value();
       double sum = 0;
       double sqr = 0;

       for(int i = 0; i < table.size(); ++i){
          sum += table[i].first;
          sqr += table[i].first * table[i].first;
       }

       double ave = sum/table.size();
       double var = sqr/table.size() - ave*ave;

       for(int i = 0; i < table.size(); ++i)
          table[i].first = (table[i].first - ave)/var;

       score_path_writer.Write(key, score);
       n_done++;
    }

    KALDI_LOG << "Finish " << n_done;

    return 0;

  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

