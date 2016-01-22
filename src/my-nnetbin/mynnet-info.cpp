#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "nnet/nnet-nnet.h"
#include "my-nnet/nnet-my-nnet.h"

using namespace kaldi;
using namespace kaldi::nnet1;
using namespace std;

int main(int argc, char *argv[]) {
  try {

    const char *usage =
        "Print human-readable information about the my neural network.\n"
        "(topology, various weight statistics, etc.) It prints to stdout.\n"
        "Usage:  nnet-info [options] <nnet-in>\n"
        "e.g.:\n"
        " nnet-info 1.nnet\n";
    
    ParseOptions po(usage);
    po.Read(argc, argv);

    if (po.NumArgs() != 1) {
      po.PrintUsage();
      exit(1);
    }

    string nnet_rxfilename = po.GetArg(1);

    MyNnet nnet;
    nnet.Read(nnet_rxfilename);

    cout << nnet.Info() << endl; 
    cout << "Number of parameters: " << nnet.NumParams() << endl;

    KALDI_LOG << "Printed info about " << nnet_rxfilename;
    return 0;
  } catch(const exception &e) {
    cerr << e.what() << '\n';
    return -1;
  }
}


