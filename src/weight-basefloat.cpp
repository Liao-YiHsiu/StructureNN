#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "fstext/fstext-lib.h"
#include "lat/kaldi-lattice.h"

using namespace std;
using namespace kaldi;

int main(int argc, char *argv[]) {
  try {

    string usage;
    usage.append("Apply y = ax1 + bx2 + ... \n")
       .append("Usage: ").append(argv[0]).append(" [options] <basefloat-wspecifier> <weight-a> <basefloat-rspecifier> [<weight-b> <basefloat-rspecifier>] ... \n")
       .append("e.g.: \n")
       .append(" ").append(argv[0]).append(" ark:out.ark 0.1 ark:feat1.ark -.5 ark:feat2.ark\n");

    ParseOptions po(usage.c_str());

    po.Read(argc, argv);

    if (po.NumArgs() < 3 || ( po.NumArgs()-3 ) % 2 != 0 ) {
      po.PrintUsage();
      exit(1);
    }

    string basefloat_wspecifier = po.GetArg(1);
    vector<double> weights;
    vector<string> basefloat_rspecifiers;

    for(int i = 2; i <po.NumArgs(); i += 2){
       double w = atof(po.GetArg(i).c_str());
       const string& rspecifier = po.GetArg(i+1); 

       weights.push_back(w);
       basefloat_rspecifiers.push_back(rspecifier);
    }

    BaseFloatWriter                   basefloat_writer(basefloat_wspecifier);
    vector<SequentialBaseFloatReader> basefloat_readers(basefloat_rspecifiers.size());

    for(int i = 0 ; i < basefloat_rspecifiers.size(); ++i)
       basefloat_readers[i].Open(basefloat_rspecifiers[i]);

    int32 n_done = 0;

    bool finish = false;
    while(!finish){

       for(int i = 0; i < basefloat_readers.size(); ++i)
          if(basefloat_readers[i].Done()){
             finish = true;
             break;
          }

       if(finish) break;

       const string &key = basefloat_readers[0].Key();
       for(int i = 1; i < basefloat_readers.size(); ++i)
          assert(key == basefloat_readers[i].Key());

       double sum = 0;
       for(int i = 0; i < basefloat_readers.size(); ++i){
          sum += weights[i] * basefloat_readers[i].Value();
       }

       basefloat_writer.Write(key, sum);
       n_done ++;
    }
    
    KALDI_LOG << "Done " << n_done << " features";
    return (n_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
