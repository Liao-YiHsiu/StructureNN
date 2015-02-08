// nnetbin/nnet-concat.cc

// Copyright 2012-2013  Brno University of Technology (Author: Karel Vesely)

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "nnet/nnet-nnet.h"

using namespace kaldi;
using namespace kaldi::nnet1;
typedef kaldi::int32 int32;

int main(int argc, char *argv[]) {
  try {

    const char *usage =
        "Pop the last few layers of Neural Networks\n"
        "Usage:  nnet-concat [options] <model-in> <model-out>\n"
        "e.g.:\n"
        " nnet-concat --binary=false --num=3 nnet.in nnet.out\n";
    
    ParseOptions po(usage);
    
    bool binary_write = true;
    int32 num = 1;
    po.Register("binary", &binary_write, "Write output in binary mode");
    po.Register("num", &num, "Pop the last n layers");

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_in_filename = po.GetArg(1);
    std::string model_out_filename = po.GetArg(2);

    //read the first nnet
    KALDI_LOG << "Reading " << model_in_filename;
    Nnet nnet; 
    {
      bool binary_read;
      Input ki(model_in_filename, &binary_read);
      nnet.Read(ki.Stream(), binary_read);
    }

    // pop the last few layers
    {
       for(int i = 0; i < num; ++i)
          nnet.RemoveLastComponent();
    }

    //finally write the nnet to disk
    {
      Output ko(model_out_filename, binary_write);
      nnet.Write(ko.Stream(), binary_write);
    }

    KALDI_LOG << "Written model to " << model_out_filename;
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}


