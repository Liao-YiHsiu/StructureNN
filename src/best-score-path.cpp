#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "svm.h"

using namespace std;
using namespace kaldi;

int main(int argc, char *argv[]) {
  try {

    string usage;
    usage.append("Pick one best path in each key by the corresponding score\n")
       .append("Usage: ").append(argv[0]).append(" [options] <score-path-rspecifier> <score-path-wspecifier>\n")
       .append(" e.g.: ").append(argv[0]).append(" ark:path_in.ark ark:path_out.ark\n");

    ParseOptions po(usage.c_str());
    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }
    srand(time(NULL));

    string score_path_rspecifier = po.GetArg(1);
    string score_path_wspecifier = po.GetArg(2);

    SequentialScorePathReader score_path_reader(score_path_rspecifier);
    ScorePathWriter           score_path_writer(score_path_wspecifier);

    for (; !score_path_reader.Done(); score_path_reader.Next()){
       const ScorePath::Table &table = score_path_reader.Value().Value();
       int index = rand()%table.size();
       BaseFloat max = table[index].first;

       for(int i = 0; i < table.size(); ++i)
          if(max < table[i].first){
             max = table[i].first;
             index = i;
          }

       ScorePath tmp;
       tmp.Value().push_back(table[index]);

       score_path_writer.Write(score_path_reader.Key(), tmp);
    }
    return 0;

  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
