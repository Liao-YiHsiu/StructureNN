#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "fstext/fstext-lib.h"
#include "lat/kaldi-lattice.h"
#include "lat/lattice-functions.h"
#include "score-path/score-path.h"
#include "my-utils/util.h"

using namespace std;
using namespace fst;
using namespace kaldi;

typedef Lattice::Arc Arc;
typedef Arc::StateId StateId;
typedef Arc::Weight Weight;

int DFS(Lattice &lat, vector<uchar> &ref_trim, vector<uchar> &path, StateId s, int plabel, int &counter){
   ArcIterator<Lattice> aiter(lat, s);

   if(aiter.Done()){
      //cout << path.size() << " ";
      //for(int i = 0; i < path.size(); ++i)
      //   cout << path[i] << " ";
      //cout << endl;
      vector<uchar> path_trim;
      trim_path(path, path_trim);
      counter++;
      return LevenshteinEditDistance(ref_trim, path_trim);
   }

   int min = ref_trim.size();
   for(; !aiter.Done(); aiter.Next()){
      const Arc &arc = aiter.Value();

      if( arc.ilabel != 0){
         if(arc.olabel == 0)
            path.push_back(plabel);
         else{
            path.push_back(arc.olabel);
            plabel = arc.olabel;
         }
      }
      int dist = DFS(lat, ref_trim, path, arc.nextstate, plabel, counter);
      if( min > dist ) 
         min = dist;

      if(arc.ilabel != 0)
         path.pop_back();

      if(min == 0)return min;
   }

   return min;
}

int main(int argc, char *argv[]) {
  try {

    string usage;
    usage.append("Traverse all path from lattice to calculate the min PER.\n")
       .append("Usage: ").append(argv[0]).append(" [options] <lats-rspecifier> <label-rspecifier>\n")
       .append(" e.g.: ").append(argv[0]).append(" ark:lat.ark ark:lab.ark \n");

    ParseOptions po(usage.c_str());

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    string lats_rspecifier = po.GetArg(1),
        label_rspecifier   = po.GetArg(2);

    // Read as regular lattice
    SequentialLatticeReader     lattice_reader(lats_rspecifier);
    SequentialUcharVectorReader label_reader(label_rspecifier);

    int N = 0;
    int err = 0;
    int num_done = 0;

    for (; !lattice_reader.Done() && !label_reader.Done(); 
          lattice_reader.Next(), label_reader.Next()) {

       assert(lattice_reader.Key() == label_reader.Key());

       const vector<uchar> &lab = label_reader.Value();
       Lattice              lat = lattice_reader.Value();
       lattice_reader.FreeCurrent();

       uint64 props = lat.Properties(kFstProperties, false);
       if (!(props & fst::kTopSorted)) {
          if (fst::TopSort(&lat) == false)
             KALDI_ERR << "Cycles detected in lattice.";
       }
       KALDI_ASSERT(lat.Start() == 0);

       vector<uchar> lab_trim;
       trim_path(lab, lab_trim);

       // do DFS search overall path
       vector<uchar> path;
       int counter = 0;
       int dist = DFS(lat, lab_trim, path, 0, -1, counter);
       err += dist;
       N += lab_trim.size();

       KALDI_LOG << ++num_done << " Finished with " << dist << " / " << lab_trim.size() << " all lats = " << counter;
    }

    return 0;

  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

