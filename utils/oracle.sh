#!/bin/bash
source path

tmp=$(mktemp)

if [ "$#" -ne 2 ]; then
   echo "Calculate the upper/lower bound of WER"
   echo "Usage: $0 <score-path-rspecifier> <label-rspecifier>"
   echo "eg. $0 \"ark:gunzip -c data/nnet/1_XXX_0_0.16.data.tag.gz |\" data/nnet/test.lab"
   echo ""
   exit 1;
fi

input=$1
label=$2

score-oracle "$label" "$input" ark:- | best-score-path ark:- ark:$tmp && \
   calc.sh "$label" ark:$tmp

score-oracle "$label" "$input" ark:- | weight-score-path ark:- -1 ark:- | \
   best-score-path ark:- ark:$tmp && calc.sh "$label" ark:$tmp

rm -rf $tmp
