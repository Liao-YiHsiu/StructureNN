// bin/post-to-phone-post.cc

// Copyright 2012-2013  Johns Hopkins University (author: Daniel Povey)

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
#include "hmm/transition-model.h"
#include "hmm/posterior.h"

using namespace std;

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;  

    const char *usage =
        "Convert posteriors to vector\n"
        "\n"
        "Usage: post-to-vec [options] <post-rspecifier> <vec-wspecifier>\n"
        " e.g.: post-to-vec ark:post.ark ark,t:-\n";
    
    ParseOptions po(usage);
    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }
      
    string post_rspecifier = po.GetArg(1),
           vec_wspecifier  = po.GetArg(2);

    SequentialPosteriorReader posterior_reader(post_rspecifier);
    Int32VectorWriter         vector_writer(vec_wspecifier);

    int num_done = 0;

    for (; !posterior_reader.Done(); posterior_reader.Next()) {
      const Posterior &posterior = posterior_reader.Value();
      vector<int32> arr(posterior.size());
      for(int i = 0; i < posterior.size(); ++i)
         arr[i] = posterior[i][0].first;

      vector_writer.Write(posterior_reader.Key(), arr);
      num_done++;
    }
    KALDI_LOG << "Done converting posteriors to vector for "
              << num_done << " utterances.";
    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

