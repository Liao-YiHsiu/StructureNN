#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include <math.h>
#include <cfloat>
#include "score-path/score-path.h"

using namespace std;
using namespace kaldi;

double Add_log(double log_a, double log_b){
   double max = log_a < log_b ? log_b:log_a;
   double min = log_a > log_b ? log_b:log_a;
   
   return max + log( 1 + exp(min - max) );
}

double Div_log(double log_a, double log_b){ 
   return log_a - log_b; 
}

double Add_no(double a, double b){ return a+b; }
double Div_no(double a, double b){ return a/b; }

int main(int argc, char *argv[]) {
  try {

    string usage;
    usage.append("Normalize the score-path to sum = 1\n")
       .append("Usage: ").append(argv[0]).append(" [options] <score-path-rspecifier> <score-path-wspecifier> \n")
       .append(" e.g.: ").append(argv[0]).append(" ark:path_in.ark ark:path_out.ark \n");

    ParseOptions po(usage.c_str());

    bool log_domain = false;
    po.Register("log-domain", &log_domain, "Input score is in log domain or not.");

    po.Read(argc, argv);

    if (po.NumArgs() != 2){
      po.PrintUsage();
      exit(1);
    }

    double (*add)(double, double) = log_domain ? Add_log: Add_no;
    double (*div)(double, double) = log_domain ? Div_log: Div_no;

    string score_path_rspecifier = po.GetArg(1);
    string score_path_wspecifier = po.GetArg(2);

    SequentialScorePathReader score_path_reader(score_path_rspecifier);
    ScorePathWriter           score_path_writer(score_path_wspecifier);

    int32 n_done = 0;
    for(;!score_path_reader.Done(); score_path_reader.Next()){
       ScorePath     score = score_path_reader.Value();
       const string& key   = score_path_reader.Key(); 

       ScorePath::Table &table = score.Value();
       double sum = log_domain ? -DBL_MAX: 0;

       for(int i = 0; i < table.size(); ++i)
          sum = add(sum, table[i].first);

       for(int i = 0; i < table.size(); ++i)
          table[i].first = div(table[i].first, sum);

       score_path_writer.Write(key, score);
       n_done++;
    }

    KALDI_LOG << "Finish " << n_done;

    return 0;

  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

