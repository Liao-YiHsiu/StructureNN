#include "base/kaldi-common.h"
#include "util/common-utils.h"

#include "my-nnet/nnet-my-nnet.h"

using namespace std;
using namespace kaldi;

int main(int argc, char *argv[]) {

   try {
      string usage;
      usage.append("My Neural Network initialization.\n")
         .append("Usage: ").append(argv[0]).append(" [options] <config-file> <nnet-out>\n")
         .append("e.g.: \n")
         .append(" ").append(argv[0]).append(" config nnet-out\n");

      ParseOptions po(usage.c_str());

      bool binary = true;
      po.Register("binary", &binary, "Write model in binary mode");

      int seed = 777;
      po.Register("seed", &seed, "Random seed for initialization");

      po.Read(argc, argv);

      if (po.NumArgs() != 2) {
         po.PrintUsage();
         exit(1);
      }

      string config_filename   = po.GetArg(1),
             nnet_out_filename = po.GetArg(2);

      srand(seed);

      MyNnet nnet;
      nnet.Init(config_filename);
      nnet.Write(nnet_out_filename, binary);

      return 0;
   } catch(const std::exception &e) {
      std::cerr << e.what();
      return -1;
   }
}

