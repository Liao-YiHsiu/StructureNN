#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "score-path/score-path.h"

using namespace std;
using namespace kaldi;

int main(int argc, char *argv[]) {
  try {

    string usage;
    usage.append("Do y = exp(x) on score-path\n")
       .append("Usage: ").append(argv[0]).append(" [options] <score-path-rspecifier> <score-path-wspecifier> \n")
       .append(" e.g.: ").append(argv[0]).append(" ark:path_in.ark ark:path_out.ark\n");

    ParseOptions po(usage.c_str());
    po.Read(argc, argv);

    if (po.NumArgs() != 2){
      po.PrintUsage();
      exit(1);
    }

    string score_path_rspecifier = po.GetArg(1);
    string score_path_wspecifier = po.GetArg(2);

    ScorePathWriter           score_path_writer(score_path_wspecifier);
    SequentialScorePathReader score_path_reader(score_path_rspecifier);

    int32 n_done = 0;
    for(; !score_path_reader.Done(); score_path_reader.Next()){
       const string       &key = score_path_reader.Key();
       ScorePath         score = score_path_reader.Value();
       ScorePath::Table &table = score.Value();

       for(int i = 0; i < table.size(); ++i)
          table[i].first = exp(table[i].first);

       score_path_writer.Write(key, table);
       n_done++;
    }
    KALDI_LOG << "Finish " << n_done;

    return 0;

  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
