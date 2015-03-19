#!/bin/bash

timit=~/Research/timit

. parse_options.sh || exit 1;

if [ "$#" -ne 3 ]; then
   echo "Use kaldi command to convert lattice to path-score"
   echo "Usage: $0 <model> <lattice-rspecifier> <path-score-wspecifier>"
   echo ""
   echo "eg. $0 final.mdl \"ark:lattice-to-nbest ark:test.lat ark:- |\" ark:test.tag"
   exit 1;
fi

model=$1
lattice=$2
path_score=$3

lattice-to-post "$lattice" ark:- |\
   post-to-phone-post "$model" ark:- ark:- |\
   post-to-vec ark:- ark,t:- | \
   $timit/utils/int2sym.pl -f 2- $timit/data/lang/phones.txt - - | \
   $timit/utils/sym2int.pl -f 2- $timit/data/lang/words.txt - | \
   vec-to-path-score ark:- "$path_score" || exit 1;
