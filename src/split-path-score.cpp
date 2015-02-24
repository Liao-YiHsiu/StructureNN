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
    usage.append("Split path score into path and score vector.\n")
       .append("Usage: ").append(argv[0]).append(" [options] <score-path-rspecifier> <score-wspecifier> <path-wspecifier>\n")
       .append("e.g.: \n")
       .append(" ").append(argv[0]).append(" ark:score_path.ark ark:score.ark ark:path.ark \n");

    ParseOptions po(usage.c_str());

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    string score_path_rspecifier  = po.GetArg(1),
      score_wspecifier            = po.GetArg(2),
      path_wspecifier             = po.GetArg(3);


    SequentialScorePathReader  score_path_reader(score_path_rspecifier);
    BaseFloatWriter           score_writer(score_wspecifier);
    Int32VectorWriter         path_writer(path_wspecifier);

    for(; !score_path_reader.Done(); score_path_reader.Next()){
       const string &key             = score_path_reader.Key();
       const ScorePath::Table &table = score_path_reader.Value().Value();

       score_writer.Write(key, table[0].first);
       path_writer.Write(key, table[0].second);
       
    }

    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}



