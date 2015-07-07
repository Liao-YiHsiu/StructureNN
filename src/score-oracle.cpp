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
    usage.append("Rescore score-path file based on FER or PER\n")
       .append("Usage: ").append(argv[0]).append(" [options] <label-rspecifier> <score-path-rspecifier> <score-path-wspecifier> \n")
       .append("e.g.: \n")
       .append(" ").append(argv[0]).append(" ark:lab.ark ark:score_path.ark ark,t:- \n");

    ParseOptions po(usage.c_str());

    string error_function = "per";
    po.Register("error-function", &error_function, "Error function : fer|per");

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    double (*acc_function)(const vector<int32>& path1, const vector<int32>& path2, double param);

    if(error_function == "fer")
       acc_function = frame_acc;
    else if(error_function == "per")
       acc_function = phone_acc; 
    else{
       po.PrintUsage();
       exit(1);
    }

    string label_rspecifier      = po.GetArg(1);
    string score_path_rspecifier = po.GetArg(2);
    string score_path_wspecifier = po.GetArg(3);

    SequentialInt32VectorReader label_reader(label_rspecifier);
    SequentialScorePathReader   score_path_reader(score_path_rspecifier);
    ScorePathWriter             score_path_writer(score_path_wspecifier);

    int numDone = 0;

    for( ; !score_path_reader.Done() && !label_reader.Done();
          score_path_reader.Next(), label_reader.Next()){

       assert(score_path_reader.Key() == label_reader.Key());

       ScorePath::Table table = score_path_reader.Value().Value();
       const vector<int32>    &ref   = label_reader.Value();

       for(int i = 0; i < table.size(); ++i){
          const vector<int32> &lab = table[i].second;
          assert(lab.size() == ref.size());

          double score = acc_function(ref, lab, 1.0);

          table[i].first = score;
       }

       score_path_writer.Write(score_path_reader.Key(), table);

       numDone++;
    }

    KALDI_LOG << "Done:" << numDone;
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}



