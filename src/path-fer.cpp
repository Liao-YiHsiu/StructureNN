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
    usage.append("Calculate Frame Error Rate via path file\n")
       .append("Usage: ").append(argv[0]).append(" [options] <path1-rspecifier> <path2-rspecifier>\n")
       .append("e.g.: \n")
       .append(" ").append(argv[0]).append(" ark:path1.ark ark:path2.ark \n");

    ParseOptions po(usage.c_str());

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    string path1_rspecifier  = po.GetArg(1),
      path2_rspecifier       = po.GetArg(2);


    SequentialUcharVectorReader    path1_reader(path1_rspecifier);
    RandomAccessUcharVectorReader  path2_reader(path2_rspecifier);
    int N = 0;
    int corrN = 0;

    for ( ; !path1_reader.Done(); path1_reader.Next()) {
       string utt = path1_reader.Key();
       assert(path2_reader.HasKey(utt));
       const vector<uchar> &arr1 = path1_reader.Value();
       const vector<uchar> &arr2 = path2_reader.Value(utt); 
       assert(arr1.size() == arr2.size());

       for(int i = 0; i < arr1.size(); ++i)
          corrN += arr1[i] == arr2[i] ? 1:0;
       N += arr1.size();
    }

    // final message
    KALDI_LOG << "Frame Error Rate = " << 1 - corrN/(double)N ;

    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


