#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "fstext/fstext-lib.h"
#include "lat/kaldi-lattice.h"
#include "lat/lattice-functions.h"
#include "util.h"

using namespace std;
using namespace fst;
using namespace kaldi;

typedef Lattice::Arc Arc;
typedef Arc::StateId StateId;
typedef Arc::Weight Weight;

void DFS(Lattice &lat, vector<int32> &path, Weight tot_weight, ScorePath &score_path, StateId s, int plabel){
   ArcIterator<Lattice> aiter(lat, s);

   if(aiter.Done()){
      cout << path.size() << " ";
      for(int i = 0; i < path.size(); ++i)
         cout << path[i] << " ";
      cout << endl;
      score_path.Value().push_back(make_pair(tot_weight.Value1() + tot_weight.Value2(), path));
      return;
   }

   for(; !aiter.Done(); aiter.Next()){
      const Arc &arc = aiter.Value();
      Weight tmp_weight = Times(arc.weight, tot_weight);

      if( arc.ilabel != 0){
         if(arc.olabel == 0)
            path.push_back(plabel);
         else{
            path.push_back(arc.olabel);
            plabel = arc.olabel;
         }
      }
      DFS(lat, path, tmp_weight, score_path, arc.nextstate, plabel);

      if(arc.ilabel != 0)
         path.pop_back();
   }
}

int main(int argc, char *argv[]) {
  try {

    string usage;
    usage.append("Use lattice to generate phone path matrix.\n")
       .append("Usage: ").append(argv[0]).append(" [options] <lats-rspecifier> <score-path-wspecifier>\n")
       .append(" e.g.: ").append(argv[0]).append(" \"ark:lattice-to-nbest ark:1.lat ark:- |\" ark:path.ark\n");

    ParseOptions po(usage.c_str());

    BaseFloat acoustic_scale = 1.0;

    po.Register("acoustic-scale", &acoustic_scale, "Scaling factor for acoustic likelihoods");
    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    string lats_rspecifier = po.GetArg(1),
        score_path_wspecifier = po.GetArg(2);

    // Read as regular lattice
    SequentialLatticeReader  lattice_reader(lats_rspecifier);
    ScorePathWriter          score_path_writer(score_path_wspecifier);
    

    for (; !lattice_reader.Done(); lattice_reader.Next()) {

       string key = lattice_reader.Key();

       Lattice lat = lattice_reader.Value();
       lattice_reader.FreeCurrent();

       fst::ScaleLattice(fst::AcousticLatticeScale(acoustic_scale), &lat);

       uint64 props = lat.Properties(kFstProperties, false);
       if (!(props & fst::kTopSorted)) {
          if (fst::TopSort(&lat) == false)
             KALDI_ERR << "Cycles detected in lattice.";
       }
       KALDI_ASSERT(lat.Start() == 0);

       // do DFS search overall path
       ScorePath score_path;
       vector<int32> path;
       DFS(lat, path, Weight::One(), score_path, 0, -1);

       score_path_writer.Write(key, score_path);
       score_path.Value().clear();
    }

    return 0;

  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
