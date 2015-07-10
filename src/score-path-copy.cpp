#include "nnet/nnet-trnopts.h"
#include "nnet/nnet-nnet.h"
#include "nnet/nnet-loss.h"
#include "nnet/nnet-randomizer.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "base/timer.h"
#include "cudamatrix/cu-device.h"
#include "svm.h"
#include <sstream>

#include <fstream>
#define MAX_MAP 10240

using namespace std;
using namespace kaldi;
using namespace kaldi::nnet1;

int main(int argc, char *argv[]) {
  
  try {
    string usage;
    usage.append("Copy Score-path.\n")
       .append("Usage: ").append(argv[0]).append(" [options] <score-path-rspecifier> <score-path-wspecifier>\n")
       .append("e.g.: \n")
       .append(" ").append(argv[0]).append(" ark:in.ark ark:out.ark \n");

    ParseOptions po(usage.c_str());

    string map_file;
    po.Register("map-file", &map_file, "map vector values according to map file");


    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    string score_path_rspecifier  = po.GetArg(1);
    string score_path_wspecifier  = po.GetArg(2);

    SequentialScorePathReader score_path_reader(score_path_rspecifier);  
    ScorePathWriter           score_path_writer(score_path_wspecifier);

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
    
    int num_done = 0;
    for(; !score_path_reader.Done(); score_path_reader.Next()){
       ScorePath score_path = score_path_reader.Value();

       if(map_file != ""){
          ScorePath::Table& table = score_path.Value();
          for(int i = 0; i < table.size(); ++i){
             vector<uchar> &arr= table[i].second;
             for(int j = 0; j < arr.size(); ++j)
                arr[j] = mapping[arr[j]];
          }
       }

       score_path_writer.Write(score_path_reader.Key(), score_path);

       num_done++;
    }

    KALDI_LOG << "Finished " << num_done;
    

    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}



