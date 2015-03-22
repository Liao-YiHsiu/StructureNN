#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "svm.h"

using namespace std;
using namespace kaldi;

int main(int argc, char *argv[]) {
  try {

    string usage;
    usage.append("Combine several vectors into a score-path file.\n")
       .append("Usage: ").append(argv[0]).append(" [options] <score-path-rspecifier> <score-path-wspecifier>\n")
       .append(" e.g.: ").append(argv[0]).append(" ark:vec.ark ark:path.ark\n");

    ParseOptions po(usage.c_str());

    string score_rspecifier;
    po.Register("score-rspecifier", score_rspecifier, "Use this file as score.");

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    string  vector_rspecifier = po.GetArg(1),
        score_path_wspecifier = po.GetArg(2);

    SequentialInt32VectorReader vector_reader(vector_rspecifier);
    ScorePathWriter             score_path_writer(score_path_wspecifier);

    SequentialBaseFloatReader   score_reader;
    if(score_rspecifier != "") score_reader.Open(score_rspecifier);
    
    ScorePath score_path;
    string pKey;

    for (; !vector_reader.Done(); vector_reader.Next()) {

       string key = vector_reader.Key();
       string nkey = key.substr(0, key.rfind('-'));

       if(score_rspecifier != "") assert(score_reader.Key() == key);

       const vector<int32> &path = vector_reader.Value();

       if(pKey == "" || nkey.compare(pKey) != 0){
          if(pKey != ""){
             score_path_writer.Write(pKey, score_path);
             score_path.Value().clear();
          }
          pKey = nkey; 
       }

       BaseFloat score = 0;
       if(score_rspecifier != "") score = score_reader.Value();
       score_path.Value().push_back(make_pair(score, path));

       if(score_rspecifier != "") score_reader.Next();
    }

    score_path_writer.Write(pKey, score_path);

    return 0;

  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
