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
    usage.append("Trim Path(Remove duplicate part of a vector)\n")
       .append("Usage: ").append(argv[0]).append(" [options] <path-rspecifier> <path-wspecifier>\n")
       .append("e.g.: \n")
       .append(" ").append(argv[0]).append(" ark:path1.ark ark:path2.ark \n");

    ParseOptions po(usage.c_str());

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    string path_rspecifier  = po.GetArg(1),
      path_wspecifier       = po.GetArg(2);


    SequentialInt32VectorReader    path_reader(path_rspecifier);
    Int32VectorWriter              path_writer(path_wspecifier);

    int N = 0;
    for ( ; !path_reader.Done(); path_reader.Next(), N++) {
       vector<int32> tmp;
       trim_path(path_reader.Value(), tmp);

       path_writer.Write(path_reader.Key(), tmp);
    }
    KALDI_LOG << "Finish " << N << " utterance.";

    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


