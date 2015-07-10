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
#include "svm.h"
#include <fstream>

#define MAX_MAP 10240

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
    
    string map_file;
    po.Register("map-file", &map_file, "map vector values according to map file");

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    int32 mapping[MAX_MAP];
    for(int i = 0; i < MAX_MAP; ++i)
       mapping[i] = i;

    if( map_file != "" ){
       ifstream fin(map_file.c_str());
       int counter = 0;
       int tmp;
       while(fin >> tmp){
          mapping[counter++] = tmp;
       }
    }
      
    string post_rspecifier = po.GetArg(1),
           vec_wspecifier  = po.GetArg(2);

    SequentialPosteriorReader posterior_reader(post_rspecifier);
    UcharVectorWriter         vector_writer(vec_wspecifier);

    int num_done = 0;

    for (; !posterior_reader.Done(); posterior_reader.Next()) {
      const Posterior &posterior = posterior_reader.Value();
      vector<uchar> arr(posterior.size());
      for(int i = 0; i < posterior.size(); ++i)
         arr[i] = mapping[posterior[i][0].first];

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

