#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "hmm/transition-model.h"
#include "transform/lda-estimate.h"
#include "svm.h"

using namespace kaldi;
using namespace std;
typedef kaldi::int32 int32;

int main(int argc, char *argv[]) {
  try {
    string usage;
    usage.append("Accumulate LDA statistics based on label.\n")
        .append("Usage: ").append(argv[0]).append(" [options] <features-rspecifier> <label-rspecifier> <lda-acc-out>\n")
        .append("e.g.\n")
        .append(" ").append(argv[0]).append(" \"ark:splice-feats scp:train.scp|\" ark:label.scp ldaacc.1\n");

    bool binary = true;
    ParseOptions po(usage.c_str());
    po.Register("binary", &binary, "Write accumulators in binary mode.");
    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string features_rspecifier = po.GetArg(1);
    std::string label_rspecifier = po.GetArg(2);
    std::string acc_wxfilename = po.GetArg(3);

    LdaEstimate lda;

    SequentialBaseFloatMatrixReader feature_reader(features_rspecifier);
    SequentialUcharVectorReader label_reader(label_rspecifier);

    vector< vector<uchar> > label_table;
    vector< string > label_key;
    uchar max_label = 0;
    // read in all label.
    for(; !label_reader.Done(); label_reader.Next()) {
       const vector<uchar> &labels = label_reader.Value();

       for(int i = 0; i < labels.size(); ++i)
          if(max_label <= labels[i]) max_label = labels[i] + 1;

       label_table.push_back(labels);
       label_key.push_back(label_reader.Key());
    }

    int32 index = 0, num_fail = 0;
    for (;!feature_reader.Done(); feature_reader.Next(), index++) {
      KALDI_ASSERT(feature_reader.Key() == label_key[index]);
      const Matrix<BaseFloat> &feats(feature_reader.Value());

      if (lda.Dim() == 0)
        lda.Init(max_label, feats.NumCols());

      if (lda.Dim() != 0 && lda.Dim() != feats.NumCols()) {
        KALDI_WARN << "Feature dimension mismatch " << lda.Dim()
                   << " vs. " << feats.NumCols();
        num_fail++;
        continue;
      }

      const vector<uchar> &label = label_table[index];
      for (int32 i = 0; i < feats.NumRows(); i++) {
        SubVector<BaseFloat> feat(feats, i);
        lda.Accumulate(feat, label[i], 1.0);
      }
      if (index % 100 == 0)
        KALDI_LOG << "Done " << index << " utterances.";
    }

    KALDI_LOG << "Done " << index << " files, failed for "
              << num_fail;

    Output ko(acc_wxfilename, binary);
    lda.Write(ko.Stream(), binary);
    KALDI_LOG << "Written statistics.";
    return (index != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


