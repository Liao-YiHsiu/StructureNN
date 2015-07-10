#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "fstext/fstext-lib.h"
#include "lat/kaldi-lattice.h"
#include "lat/lattice-functions.h"
#include "svm.h"

using namespace std;
using namespace fst;
using namespace kaldi;

int main(int argc, char *argv[]) {
  try {
    typedef Lattice::Arc Arc;
    typedef Arc::StateId StateId;
    typedef Arc::Weight Weight;

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
    
    ScorePath score_path;
    string pKey;

    for (; !lattice_reader.Done(); lattice_reader.Next()) {

       string key = lattice_reader.Key();
       string nkey = key.substr(0, key.rfind('-'));

       Lattice lat = lattice_reader.Value();
       // FreeCurrent() is an optimization that prevents the lattice from being
       // copied unnecessarily (OpenFst does copy-on-write).
       lattice_reader.FreeCurrent();

       fst::ScaleLattice(fst::AcousticLatticeScale(acoustic_scale), &lat);

       uint64 props = lat.Properties(kFstProperties, false);
       if (!(props & fst::kTopSorted)) {
          if (fst::TopSort(&lat) == false)
             KALDI_ERR << "Cycles detected in lattice.";
       }
       KALDI_ASSERT(lat.Start() == 0);

       vector<uchar> path;

       // DFS
       StateId s = 0;

       Weight tot_weight = Weight::One();

       int plabel = 0;
       while( true ){
          ArcIterator<Lattice> aiter(lat, s);
          if(aiter.Done()) break;

          const Arc &arc = aiter.Value();
          tot_weight = Times(arc.weight, tot_weight);

          if( arc.ilabel != 0){
             if(arc.olabel == 0)
                path.push_back(plabel);
             else{
                path.push_back(arc.olabel);
                plabel = arc.olabel;
             }
          }
          s = arc.nextstate;
       }

       if(pKey == "" || nkey.compare(pKey) != 0){
          if(pKey != ""){
             score_path_writer.Write(pKey, score_path);
             score_path.Value().clear();
          }
          pKey = nkey; 
       }

       score_path.Value().push_back(make_pair(
                tot_weight.Value1() + tot_weight.Value2(),
                path
                ));
    }

    score_path_writer.Write(pKey, score_path);

    return 0;

  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
