#include "nnet/nnet-trnopts.h"
#include "nnet/nnet-nnet.h"
#include "nnet/nnet-loss.h"
#include "nnet/nnet-randomizer.h"
#include "nnet/nnet-randomizer.cc"  // for template class
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "base/timer.h"
#include "cudamatrix/cu-device.h"
#include "util.h"
#include "srnnet2.h"
#include <sstream>
#include <omp.h>

using namespace std;
using namespace kaldi;

int main(int argc, char *argv[]) {
  
  try {
    string usage;
    usage.append("Naiive guess from label seq.\n")
       .append("Usage: ").append(argv[0]).append(" [options] <label-rspecifier>\n")
       .append("e.g.: \n")
       .append(" ").append(argv[0]).append(" ark:lab.ark \n");

    ParseOptions po(usage.c_str());
    po.Read(argc, argv);

    if (po.NumArgs() != 1) {
      po.PrintUsage();
      exit(1);
    }

    // setup input parameters
    srand(time(NULL));

    string label_rspecifier = po.GetArg(1);

    SequentialUcharVectorReader label_reader(label_rspecifier);

    int N = 0;
    int correct = 0;

    for ( ; !label_reader.Done(); label_reader.Next()) {

       const vector<uchar>     &label = label_reader.Value();

       uchar prev = 38;
       for(int i = 0; i < label.size(); ++i){
          if(label[i] == prev)
             correct++;
          else if(rand() % 48 + 1 == label[i])
             correct++;
          prev = label[i];
       }

       N += label.size();
    } 

    KALDI_LOG << "Frame Acc = " << correct * 100.0 / N << " %.";

    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }

}

