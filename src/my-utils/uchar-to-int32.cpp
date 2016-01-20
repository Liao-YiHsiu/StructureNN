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
    usage.append("Transform uchar vector into int32 vector")
       .append("Usage: ").append(argv[0]).append(" [options] <uchar-rspecifier> <int32-wspecifier>\n")
       .append("e.g.: \n")
       .append(" ").append(argv[0]).append(" ark:label.ark ark:label32.ark \n");

    ParseOptions po(usage.c_str());

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    string vector_rspecifier  = po.GetArg(1),
      vector_wspecifier       = po.GetArg(2);


    SequentialUcharVectorReader vector_reader(vector_rspecifier);
    Int32VectorWriter           vector_writer(vector_wspecifier);

    int N = 0;
    for ( ; !vector_reader.Done(); vector_reader.Next(), N++) {
       vector<int32> tmp;

       UcharToInt32(vector_reader.Value(), tmp);

       vector_writer.Write(vector_reader.Key(), tmp);
    }
    KALDI_LOG << "Finish " << N << " utterance.";

    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


