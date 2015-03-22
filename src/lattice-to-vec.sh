#!/bin/bash

timit=~/Research/timit
cpus=4

dir=$(mktemp -d)

. parse_options.sh || exit 1;

if [ "$#" -ne 3 ]; then
   echo "Use kaldi command to convert lattice to score-path"
   echo "Usage: $0 <model> <lattice-rspecifier> <score-path-wspecifier>"
   echo ""
   echo "eg. $0 final.mdl \"ark:lattice-to-nbest ark:test.lat ark:- |\" ark:test.tag"
   exit 1;
fi

model=$1
lattice=$2
score_path=$3

lattice-copy "$lattice" ark,scp:$dir/tmp.ark,$dir/tmp.scp || exit 1;
for (( i=0 ; i<cpus ; i++ ))
do
   file_in=$dir/in_${i}.scp
   file_out_scp=$dir/out_${i}.scp
   file_out_ark=$dir/out_${i}.ark

   $timit/utils/split_scp.pl -j $cpus $i $dir/tmp.scp $file_in || exit 1;
   lattice-to-post scp:$file_in ark,scp:$file_out_ark,$file_out_scp & 

done

for job in `jobs -p`
do
   wait $job || exit 1; 
done

file_out=$dir/out.scp
#combine jobs
for (( i=0 ; i<cpus ; i++ ))
do
   file_out_scp=$dir/out_${i}.scp
   cat $file_out_scp >> $file_out
done

   post-to-phone-post "$model" scp:$file_out ark:- |\
   post-to-vec ark:- ark,t:- | \
   $timit/utils/int2sym.pl -f 2- $timit/data/lang/phones.txt - - | \
   $timit/utils/sym2int.pl -f 2- $timit/data/lang/words.txt - | \
   vec-to-score-path ark:- "$score_path" || exit 1;

rm -rf $dir
