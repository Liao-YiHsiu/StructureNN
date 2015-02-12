#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "fstext/fstext-lib.h"
#include "lat/kaldi-lattice.h"
#include "lat/lattice-functions.h"

using namespace std;
using namespace fst;
using namespace kaldi;

int main(int argc, char *argv[]) {
  try {
    typedef Lattice::Arc Arc;
    typedef Arc::StateId StateId;

    string usage;
    usage.append("Use lattice to generate phone path matrix.\n")
       .append("Usage: ").append(argv[0]).append(" [options] <lats-rspecifier> <path-wspecifier>\n")
       .append(" e.g.: ").append(argv[0]).append(" \"ark:lattice-to-nbest ark:1.lat ark:- |\" ark:path.ark\n");

    ParseOptions po(usage.c_str());
    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    string lats_rspecifier = po.GetArg(1),
        path_wspecifier = po.GetArg(2);

    // Read as regular lattice
    SequentialLatticeReader lattice_reader(lats_rspecifier);

    Int32VectorVectorWriter path_writer(path_wspecifier);
    
    vector< vector<int32> > path_matrix;
    string pKey;

    for (; !lattice_reader.Done(); lattice_reader.Next()) {

       string key = lattice_reader.Key();
       Lattice lat = lattice_reader.Value();
       // FreeCurrent() is an optimization that prevents the lattice from being
       // copied unnecessarily (OpenFst does copy-on-write).
       lattice_reader.FreeCurrent();

       uint64 props = lat.Properties(kFstProperties, false);
       if (!(props & fst::kTopSorted)) {
          if (fst::TopSort(&lat) == false)
             KALDI_ERR << "Cycles detected in lattice.";
       }
       KALDI_ASSERT(lat.Start() == 0);

       vector<int32> path;

       //int32 num_states = lat.NumStates();
       //vector<int32> state_times;
       //int32 max_time = LatticeStateTimes(lat, &state_times);

       // DFS
       StateId s = 0;
       int plabel;
       while( true ){
          ArcIterator<Lattice> aiter(lat, s);
          if(aiter.Done()) break;

          const Arc &arc = aiter.Value();
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

       if(pKey == "" || key.compare(0, pKey.size(), pKey) != 0){
          if(pKey != ""){
             path_writer.Write(pKey, path_matrix);
             path_matrix.clear();
          }
          pKey = key.substr(0, key.rfind('-'));
       }

       path_matrix.push_back(path);
    }

    path_writer.Write(pKey, path_matrix);

    return 0;

  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
