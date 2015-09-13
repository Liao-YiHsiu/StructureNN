#include "nnet/nnet-trnopts.h"
#include "nnet/nnet-nnet.h"
#include "nnet/nnet-loss.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "base/timer.h"
#include "cudamatrix/cu-device.h"
#include "util.h"
#include "srnnet.h"
#include "kernel.h"
#include <sstream>
#include <pthread.h>



using namespace std;
using namespace kaldi;
using namespace kaldi::nnet1;

int main(int argc, char *argv[]) {

   try {
      string usage;
      usage.append("Structure Recurrent Neural Network initialization.\n")
         .append("Usage: ").append(argv[0]).append(" [options] <nnet1-in> <nnet2-in> <config-file> <nnet-out>\n")
         .append("e.g.: \n")
         .append(" ").append(argv[0]).append(" nnet1 nnet2 config nnet-out\n");

      ParseOptions po(usage.c_str());

      bool binary = true;
      po.Register("binary", &binary, "Write model in binary mode");

      po.Read(argc, argv);

      if (po.NumArgs() != 4) {
         po.PrintUsage();
         exit(1);
      }

      string nnet1_in_filename = po.GetArg(1),
             nnet2_in_filename = po.GetArg(2),
             config_filename   = po.GetArg(3),
             nnet_out_filename = po.GetArg(4);

      Nnet nnet1, nnet2;

      nnet1.Read(nnet1_in_filename);
      nnet2.Read(nnet2_in_filename);

      SRNnet nnet;
      nnet.Init(nnet1, nnet2, config_filename);
      nnet.Write(nnet_out_filename, binary);

      return 0;
   } catch(const std::exception &e) {
      std::cerr << e.what();
      return -1;
   }
}

