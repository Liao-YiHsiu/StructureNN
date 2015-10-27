#!/bin/bash -ex
source path

acwt=0.16
test_lattice_N=10
cpus=$(nproc)
lat_model=$timit/exp/dnn4_pretrain-dbn_dnn_smbr/final.mdl
tmpdir=$(mktemp -d)

echo "$0 $@"  # Print the command line for logging

. parse_options.sh || exit 1;

if [ "$#" -ne 1 ]; then
   echo "Output Kaldi baseline"
fi

   test_lattice_path=data/lab/test.lab_${test_lattice_N}_${acwt}.gz
   lattice-to-nbest-path.sh --cpus $cpus --acoustic-scale $acwt --n $test_lattice_N \
      $lat_model ark:data/test.lat "$test_lattice_path" 

   weight-score-path ark:- -1.0 "ark:gunzip -c $test_lattice_path |" | \
      best-score-path ark:- ark:$tmpdir/test.ark

   calc.sh ark:data/test.lab ark:$tmpdir/test.ark

   rm -rf $tmpdir
