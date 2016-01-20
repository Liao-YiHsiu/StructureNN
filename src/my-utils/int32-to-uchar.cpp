#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "score-path/score-path.h"
#include "my-utils/util.h"
#include <sstream>

using namespace std;
using namespace kaldi;

int main(int argc, char *argv[]) {
  
  try {
    string usage;
    usage.append("Transform int32 vector into uchar vector")
       .append("Usage: ").append(argv[0]).append(" [options] <int32-rspecifier> <uchar-wspecifier>\n")
       .append("e.g.: \n")
       .append(" ").append(argv[0]).append(" ark:label32.ark ark:label.ark \n");

    ParseOptions po(usage.c_str());

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    string vector_rspecifier  = po.GetArg(1),
      vector_wspecifier       = po.GetArg(2);


    SequentialInt32VectorReader vector_reader(vector_rspecifier);
    UcharVectorWriter           vector_writer(vector_wspecifier);

    int N = 0;
    for ( ; !vector_reader.Done(); vector_reader.Next(), N++) {
       vector<uchar> tmp;

       Int32ToUchar(vector_reader.Value(), tmp);

       vector_writer.Write(vector_reader.Key(), tmp);
    }
    KALDI_LOG << "Finish " << N << " utterance.";

    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


