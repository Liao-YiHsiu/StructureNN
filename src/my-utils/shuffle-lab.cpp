#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "my-utils/util.h"
#include "score-path/score-path.h"

using namespace std;
using namespace kaldi;

int main(int argc, char *argv[]) {
  
  try {
    string usage;
    usage.append("Shuffle labels.\n")
       .append("Usage: ").append(argv[0]).append(" [options] <label-rspecifier> <label-wspecifier>\n")
       .append("e.g.: \n")
       .append(" ").append(argv[0]).append(" ark:in.ark ark:out.ark \n");

    ParseOptions po(usage.c_str());

    int rand_seed = 777;
    po.Register("rand-seed", &rand_seed, "shuffling random seed");

    int batch_size = 512;
    po.Register("batch-size", &batch_size, "shuffling batch size");

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    string rspecifier  = po.GetArg(1);
    string wspecifier  = po.GetArg(2);

    SequentialUcharVectorReader reader(rspecifier);  
    UcharVectorWriter           writer(wspecifier);

    vector<string>             keys(batch_size); 
    vector< vector<uchar> >    values(batch_size);

    int num_done = 0;
    srand(rand_seed);
    while(1){
       int num;

       // filled in randomizer
       for(num = 0; num < batch_size && !reader.Done(); 
             ++num, reader.Next()){
          keys[num]   = reader.Key();
          values[num] = reader.Value();
       }

       if(num == 0)break;

       vector<int> shfIdx(num);
       for(int i = 0; i < num; ++i)
          shfIdx[i] = i;
       for(int i = 0; i < num; ++i){
          int t = rand() % num;

          //swap shfIdx[i] and shfIdx[t]
          int tmp = shfIdx[i];
          shfIdx[i] = shfIdx[t];
          shfIdx[t] = tmp;
       }

       for(int i = 0; i < num; ++i){
          writer.Write(keys[shfIdx[i]], values[shfIdx[i]]);
       }

       num_done += num;
    }

    KALDI_LOG << "Finished " << num_done;
    

    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}



