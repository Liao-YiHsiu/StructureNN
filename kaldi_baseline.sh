#!/bin/bash
source path

tmp=$(mktemp)

if [ "$#" -ne 2 ]; then
   echo "Evaluate kaldi-baseline"
   echo "Usage: $0 <score-path-rspecifier> <label-rspecifier>"
   echo "eg. $0 \"ark:gunzip -c data/nnet/train.lab_XXX_0.16.gz |\" data/nnet/train.lab"
   echo ""
   exit 1;
fi

input=$1
label=$2

weight-score-path ark:- -1 "$input" | best-score-path ark:- ark:$tmp && \
   calc.sh "$label" ark:$tmp

rm -rf $tmp
