#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "score-path/score-path.h"

using namespace std;
using namespace kaldi;

int main(int argc, char *argv[]) {
  try {

    string usage;
    usage.append("Do y = ax1 + bx2 + cx3 +... on score-path\n")
       .append("Usage: ").append(argv[0]).append(" [options] <score-path-wspecifier> <weight-a> <score-path-rspecifier> [<weight-b> <score-path-rspecifier>] ... \n")
       .append(" e.g.: ").append(argv[0]).append(" ark:path_out.ark 1.0 ark:path1.ark 0.3 ark:path2.ark\n");

    ParseOptions po(usage.c_str());
    po.Read(argc, argv);

    if (po.NumArgs() < 3 || ( po.NumArgs()-3 ) % 2 != 0 ) {
      po.PrintUsage();
      exit(1);
    }

    string score_path_wspecifier = po.GetArg(1);
    vector<double> weights;
    vector<string> score_path_rspecifiers;

    for(int i = 2; i <po.NumArgs(); i += 2){
       double w = atof(po.GetArg(i).c_str());
       const string& rspecifier = po.GetArg(i+1); 

       weights.push_back(w);
       score_path_rspecifiers.push_back(rspecifier);
    }

    ScorePathWriter                   score_path_writer(score_path_wspecifier);
    vector<SequentialScorePathReader> score_path_readers(score_path_rspecifiers.size());

    for(int i = 0 ; i < score_path_rspecifiers.size(); ++i)
       score_path_readers[i].Open(score_path_rspecifiers[i]);

    int32 n_done = 0;

    bool finish = false;
    while(!finish){
       for(int i = 0; i < score_path_readers.size(); ++i)
          if(score_path_readers[i].Done()){
             finish = true;
             break;
          }

       if(finish) break;

       const string &key = score_path_readers[0].Key();
       for(int i = 1; i < score_path_readers.size(); ++i)
          assert(key == score_path_readers[i].Key());

       // check arr
       const ScorePath::Table &table = score_path_readers[0].Value().Value();
       for(int i = 1; i < score_path_readers.size(); ++i){
          const ScorePath::Table &now_table = score_path_readers[i].Value().Value();

          assert(table.size() == now_table.size());
          for(int j = 0; j < table.size(); ++j){
             // check table[i].second with now_table[i].second
             const vector<uchar> &arr = table[i].second;
             const vector<uchar> &now_arr = now_table[i].second;
             assert(arr.size() == now_arr.size());
             for(int k = 0; k < now_arr.size(); ++k)
                assert(arr[k] == now_arr[k]);
          }
       }

       ScorePath::Table out = table;
       for(int i = 0; i < out.size(); ++i)
          out[i].first = 0;
       
       for(int i = 0; i < score_path_readers.size(); ++i){
          const ScorePath::Table& now_table = score_path_readers[i].Value().Value();

          for(int j = 0; j < now_table.size(); ++j){
             out[j].first += weights[i] * now_table[j].first;
          }
       }


       score_path_writer.Write(key, out);
       n_done ++;

       for(int i = 0; i < score_path_readers.size(); ++i)
          score_path_readers[i].Next();
    }

    KALDI_LOG << "Finish " << n_done;

    return 0;

  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
