#!/bin/bash

timit=~/Research/timit
map_file=~/Research/src/mapping.txt

config=
help=
print_args=
verbose=

acoustic_scale=
n=
random=
srand=
cpus=2

dir=$(mktemp -d)

echo "$0 $@" >&2  # Print the command line for logging

. parse_options.sh || exit 1;

if [ "$#" -ne 3 ]; then
   echo "a wrapper script for lattice-to-nbest and lattice-to-vec for mulitple cpu computation."
   echo "Usage: $0 [options] <model> <lattice-rspecifier> <score-path-wspecifier>"
   echo "e.g.: $0 --cpus 2 --acoustic-scale 0.1 --n 10 final.mdl ark:1.lats ark:nbest.ark"
   echo ""
   exit 1;
fi

model=$1
lattice_r=$2
path_score=$3

if [ $cpus == 1 ]; then
   echo "Not for single cpu"
   exit 1;
fi


lattice-copy "$lattice_r" ark,scp:$dir/tmp.ark,$dir/tmp.scp || exit 1;

for (( i=0 ; i<cpus ; i++ ))
do
   file_in=$dir/in_${i}.scp
   file_out_scp=$dir/out_${i}.scp
   file_out_ark=$dir/out_${i}.ark

   $timit/utils/split_scp.pl -j $cpus $i $dir/tmp.scp $file_in || exit 1;
   lattice-to-nbest \
      ${config:+ --config=$config} \
      ${help:+ --help=$help} \
      ${print_args:+ --print-args=$print_args} \
      ${verbose:+ --verbose=$verbose} \
      ${acoustic_scale:+ --acoustic-scale=$acoustic_scale} \
      ${n:+ --n=$n} \
      ${random:+ --random=$random} \
      ${srand:+ --srand=$srand} \
      scp:$file_in ark:- |\
   lattice-to-post ark:- ark,scp:$file_out_ark,$file_out_scp & 

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

   post-to-phone-post "$model" scp:$file_out ark:- | \
   post-to-vec --map-file=$map_file ark:- ark:- | \
   vec-to-path-score ark:- "$path_score" || exit 1;
#   $timit/utils/int2sym.pl -f 2- $timit/data/lang/phones.txt - - | \
#   $timit/utils/sym2int.pl -f 2- $timit/data/lang/words.txt - | \
#   vec-to-path-score ark:- "$path_score" || exit 1;

rm -rf $dir
