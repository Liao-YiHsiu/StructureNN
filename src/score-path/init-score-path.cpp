#include "nnet/nnet-trnopts.h"
#include "nnet/nnet-nnet.h"
#include "nnet/nnet-loss.h"
#include "nnet/nnet-randomizer.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "base/timer.h"
#include "cudamatrix/cu-device.h"
#include "util.h"
#include <sstream>

using namespace std;
using namespace kaldi;
using namespace kaldi::nnet1;

int main(int argc, char *argv[]) {
  
  try {
    string usage;
    usage.append("Initial Score-path file from feature file.\n")
       .append("Usage: ").append(argv[0]).append(" [options] <feature-rspecifier> <score-path-wspecifier> \n")
       .append("e.g.: \n")
       .append(" ").append(argv[0]).append(" ark:feat.ark ark:path.ark\n");

    ParseOptions po(usage.c_str());

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }
    string feature_rspecifier    = po.GetArg(1);
    string score_path_wspecifier = po.GetArg(2);


    SequentialBaseFloatMatrixReader   feature_reader(feature_rspecifier);
    ScorePathWriter                   score_path_writer(score_path_wspecifier);

    ScorePath empty;

    for(; !feature_reader.Done(); feature_reader.Next()){
       score_path_writer.Write(feature_reader.Key(), empty);
    }

    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

