#!/bin/bash
source path

tmpdir=$(mktemp -d)

if [ "$#" -ne 2 ]; then
   echo "Convert score-path zip results into a python readable file"
   echo "Usage: $0 <score-path-zipfile> <outfile>"
   echo "eg. $0 data/nnet/1_10_2_400_0.0001_0.16.data.tag.gz out"
   echo ""
   exit 1;
fi

zipfile=$1
outfile=$2

dir=$(dirname $zipfile)

pred_score="ark:gunzip -c $zipfile |"
real_score="ark:gunzip -c $tmpdir/real.tgz.gz |"

   # PER or FER of a sentence.
   score-oracle ark:$dir/test.lab "$pred_score" "ark:| gzip -c > $tmpdir/real.tgz.gz" 

   # generating the points.
   score-path-point "$pred_score" "$real_score" > $outfile

#rm -rf $tmpdir
