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
    usage.append("Convert SVM out to int32 vector using label to re-align\n")
       .append("Usage: ").append(argv[0]).append(" [options] <svm-tags> <label-rspecifier> <path-wspecifier>\n")
       .append("e.g.: \n")
       .append(" ").append(argv[0]).append(" test.tags ark:test.lab ark:path.ark \n");

    ParseOptions po(usage.c_str());

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    string svm_tags_file = po.GetArg(1),
      label_rspecifier   = po.GetArg(2),
      path_wspecifier    = po.GetArg(3);


    ifstream  fin(svm_tags_file.c_str());
    SequentialUcharVectorReader  label_reader(label_rspecifier);
    UcharVectorWriter            path_writer(path_wspecifier);

    int N = 0;
    for ( ; !label_reader.Done(); label_reader.Next(), N++) {

       const vector<uchar> &label = label_reader.Value(); 

       vector<uchar> tmp(label.size());
       for(int i = 0; i < label.size(); ++i)
          assert(fin >> tmp[i]);

       path_writer.Write(label_reader.Key(), tmp);
    }

    // final message
    KALDI_LOG << "Finish " << N << " utterances.";

    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


