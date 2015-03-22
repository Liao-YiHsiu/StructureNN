#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "fstext/fstext-lib.h"
#include "lat/kaldi-lattice.h"

using namespace kaldi;

int main(int argc, char *argv[]) {
  try {

    string usage;
    usage.append("Apply y = ax1 + bx2 + ... \n")
       .append("Usage: ").append(argv[0]).append(" [options] <base-float-wspecifier> [<weight-a> <basefloat-rspecifier>] [<weight-b> <basefloat-rspecifier>] ... \n")
       .append("e.g.: \n")
       .append(" ").append(argv[0]).append(" ark:out.ark ark:path1.ark ark:path2.ark\n");

    ParseOptions po(usage.c_str());

    po.Read(argc, argv);

    if (po.NumArgs() < 2 || po.NumArgs() > 5) {
      po.PrintUsage();
      exit(1);
    }

    std::string lats_rspecifier = po.GetArg(1),
        ali_wspecifier = po.GetArg(2),
        trans_wspecifier = po.GetOptArg(3),
        lm_cost_wspecifier = po.GetOptArg(4),
        ac_cost_wspecifier = po.GetOptArg(5);

    SequentialLatticeReader lattice_reader(lats_rspecifier);

    Int32VectorWriter ali_writer(ali_wspecifier);
    Int32VectorWriter trans_writer(trans_wspecifier);
    BaseFloatWriter lm_cost_writer(lm_cost_wspecifier);
    BaseFloatWriter ac_cost_writer(ac_cost_wspecifier);
    
    int32 n_done = 0, n_err = 0;
    
    for (; !lattice_reader.Done(); lattice_reader.Next()) {
      std::string key = lattice_reader.Key();
      Lattice lat = lattice_reader.Value();

      vector<int32> ilabels;
      vector<int32> olabels;
      LatticeWeight weight;
      
      if (!GetLinearSymbolSequence(lat, &ilabels, &olabels, &weight)) {
        KALDI_WARN << "Lattice/nbest for key " << key << " had wrong format: "
            "note, this program expects input with one path, e.g. from "
            "lattice-to-nbest.";
        n_err++;
      } else {
        if (ali_wspecifier != "") ali_writer.Write(key, ilabels);
        if (trans_wspecifier != "") trans_writer.Write(key, olabels);
        if (lm_cost_wspecifier != "") lm_cost_writer.Write(key, weight.Value1());
        if (ac_cost_wspecifier!= "") ac_cost_writer.Write(key, weight.Value2());
        n_done++;
      }
    }
    KALDI_LOG << "Done " << n_done << " n-best entries, "
              << n_err  << " had errors.";
    return (n_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
