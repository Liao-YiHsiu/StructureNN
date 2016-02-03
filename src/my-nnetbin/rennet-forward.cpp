// nnetbin/nnet-forward.cc

// Copyright 2011-2013  Brno University of Technology (Author: Karel Vesely)

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

#include <limits>

#include "nnet/nnet-nnet.h"
#include "nnet/nnet-loss.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "base/timer.h"
#include "my-nnet/nnet-my-rennet.h"
#include "my-cumatrix/cu-matrix.h"


using namespace kaldi;
using namespace kaldi::nnet1;
using namespace std;
typedef kaldi::int32 int32;

int main(int argc, char *argv[]) {
   try {
      const char *usage =
         "Perform forward pass through Recurrent Ensemble Neural Network.\n"
         "\n"
         "Usage:  rennet-forward [options] <model-in> <feature-rspecifier> <feature-wspecifier>\n"
         "e.g.: \n"
         " rennet-forward nnet ark:features.ark ark:mlpoutput.ark\n";

      ParseOptions po(usage);

      string feature_transform;
      po.Register("feature-transform", &feature_transform, "Feature transform in front of main network (in nnet format)");

      string use_gpu="no";
      po.Register("use-gpu", &use_gpu, "yes|no|optional, only has effect if compiled with CUDA"); 

      int32 depth = 10;
      po.Register("depth", &depth, "the depth of Recurrent");

      po.Read(argc, argv);

      if (po.NumArgs() != 3) {
         po.PrintUsage();
         exit(1);
      }

      string model_filename = po.GetArg(1),
         feature_rspecifier = po.GetArg(2),
         feature_wspecifier = po.GetArg(3);

      //Select the GPU
#if HAVE_CUDA==1
      CuDevice::Instantiate().SelectGpuId(use_gpu);
#endif

      Nnet nnet_transf;
      if (feature_transform != "") {
         nnet_transf.Read(feature_transform);
      }

      RENnet nnet;
      nnet.Read(model_filename);
      nnet.SetDepth(depth);

      // disable dropout,
      nnet_transf.SetDropoutRetention(1.0);
      nnet.SetDropoutRetention(1.0);

      kaldi::int64 tot_t = 0;

      SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
      BaseFloatMatrixWriter feature_writer(feature_wspecifier);

      CuMatrix<BaseFloat> feats, feats_transf;
      MyCuMatrix<BaseFloat> nnet_out;
      Matrix<BaseFloat> nnet_out_host;


      Timer time;
      double time_now = 0;
      int32 num_done = 0;
      // iterate over all feature files
      for (; !feature_reader.Done(); feature_reader.Next()) {
         // read
         Matrix<BaseFloat> mat = feature_reader.Value();
         string utt = feature_reader.Key();
         KALDI_VLOG(2) << "Processing utterance " << num_done+1 
            << ", " << utt
            << ", " << mat.NumRows() << "frm";


         if (!KALDI_ISFINITE(mat.Sum())) { // check there's no nan/inf,
            KALDI_ERR << "NaN or inf found in features for " << utt;
         }


         // push it to gpu,
         feats = mat;

         // fwd-pass, feature transform,
         nnet_transf.Feedforward(feats, &feats_transf);
         if (!KALDI_ISFINITE(feats_transf.Sum())) { // check there's no nan/inf,
            KALDI_ERR << "NaN or inf found in transformed-features for " << utt;
         }

         // fwd-pass, nnet,
         nnet.Feedforward(feats_transf, &nnet_out);
         if (!KALDI_ISFINITE(nnet_out.Sum())) { // check there's no nan/inf,
            KALDI_ERR << "NaN or inf found in nn-output for " << utt;
         }

         // download from GPU,
         nnet_out_host.Resize(nnet_out.NumRows(), nnet_out.NumCols());
         nnet_out.CopyToMat(&nnet_out_host);


         // write,
         if (!KALDI_ISFINITE(nnet_out_host.Sum())) { // check there's no nan/inf,
            KALDI_ERR << "NaN or inf found in final output nn-output for " << utt;
         }
         feature_writer.Write(feature_reader.Key(), nnet_out_host);

         // progress log
         if (num_done % 100 == 0) {
            time_now = time.Elapsed();
            KALDI_VLOG(1) << "After " << num_done << " utterances: time elapsed = "
               << time_now/60 << " min; processed " << tot_t/time_now
               << " frames per second.";
         }
         num_done++;
         tot_t += mat.NumRows();
      }

      // final message
      KALDI_LOG << "Done " << num_done << " files" 
         << " in " << time.Elapsed()/60 << "min," 
         << " (fps " << tot_t/time.Elapsed() << ")"; 

#if HAVE_CUDA==1
      if (kaldi::g_kaldi_verbose_level >= 1) {
         CuDevice::Instantiate().PrintProfile();
      }
#endif

      if (num_done == 0) return -1;
      return 0;
   } catch(const exception &e) {
      cerr << e.what();
      return -1;
   }
}
