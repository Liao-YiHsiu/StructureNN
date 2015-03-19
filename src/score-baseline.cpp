#include "nnet/nnet-trnopts.h"
#include "nnet/nnet-nnet.h"
#include "nnet/nnet-loss.h"
#include "nnet/nnet-randomizer.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "base/timer.h"
#include "cudamatrix/cu-device.h"
#include "util/edit-distance.h"
#include "svm.h"
#include <sstream>

using namespace std;
using namespace kaldi;
using namespace kaldi::nnet1;

int main(int argc, char *argv[]) {
  
  try {
    string usage;
    usage.append("Calculate a batch of Score-path FER or PER\n")
       .append("Output average, max, min score for FER or PER\n")
       .append("Usage: ").append(argv[0]).append(" [options] <label-rspecifier> <score-path-rspecifier> \n")
       .append("e.g.: \n")
       .append(" ").append(argv[0]).append(" ark:lab.ark ark:score_path.ark\n");

    ParseOptions po(usage.c_str());

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    string label_rspecifier      = po.GetArg(1);
    string score_path_rspecifier = po.GetArg(2);

    SequentialScorePathReader   score_path_reader(score_path_rspecifier);
    SequentialInt32VectorReader label_reader(label_rspecifier);

    int aveCnt = 0,
        aveN   = 0,
        maxCnt = 0,
        minCnt = 0,
        N = 0;

    for( ; !score_path_reader.Done() && !label_reader.Done();
          score_path_reader.Next(), label_reader.Next()){

       assert(score_path_reader.Key() == label_reader.Key());
       const ScorePath::Table &table = score_path_reader.Value().Value();
       const vector<int32>    &ref   = label_reader.Value();

       vector<int32> ref_trim;
       trim_path(ref, ref_trim);

       int max = 0;
       int min = ref.size();

       for(int i = 0; i < table.size(); ++i){
          // compare table[i].second with label
          const vector<int32> &lab = table[i].second;
          assert(lab.size() == ref.size());

          vector<int32> lab_trim;
          trim_path(lab, lab_trim);

          int32 dist = LevenshteinEditDistance(ref_trim, lab_trim);
          
          aveCnt += dist;
          aveN += ref_trim.size();

          if(max < dist) max = dist;
          if(min > dist) min = dist;
       }
       maxCnt += max;
       minCnt += min;
       N += ref_trim.size();
    }

    KALDI_LOG << "Average Error Rate:" << aveCnt / (double) aveN;
    KALDI_LOG << "Max Error Rate:" << maxCnt / (double) N;
    KALDI_LOG << "Min Error Rate:" << minCnt / (double) N;
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}



