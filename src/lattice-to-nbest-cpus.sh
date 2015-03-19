#!/bin/bash

timit_root=~/Research/timit

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

if [ "$#" -ne 2 ]; then
   echo "a wrapper script for lattice-to-nbest for mulitple cpu computation."
   echo "Usage: $0 [options] lattice-rspecifier lattice-wspecifier"
   echo "e.g.: $0 --cpus 2 --acoustic-scale 0.1 --n 10 ark:1.lats ark:nbest.lats"
   echo ""
   exit 1;
fi

lattice_r=$1
lattice_w=$2

if [ $cpus == 1 ]; then
   echo "Not for single cpu"
   exit 1;
fi


lattice-copy "$1" ark,scp:$dir/tmp.ark,$dir/tmp.scp || exit 1;

for (( i=0 ; i<cpus ; i++ ))
do
   file_in=$dir/in_${i}.scp
   file_out_scp=$dir/out_${i}.scp
   file_out_ark=$dir/out_${i}.ark

   $timit_root/utils/split_scp.pl -j $cpus $i $dir/tmp.scp $file_in || exit 1;
   lattice-to-nbest \
      ${config:+ --config=$config} \
      ${help:+ --help=$help} \
      ${print_args:+ --print-args=$print_args} \
      ${verbose:+ --verbose=$verbose} \
      ${acoustic_scale:+ --acoustic-scale=$acoustic_scale} \
      ${n:+ --n=$n} \
      ${random:+ --random=$random} \
      ${srand:+ --srand=$srand} \
      scp:$file_in ark,scp:$file_out_ark,$file_out_scp & 

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
lattice-copy scp:$file_out "$2" || exit 1;
rm -rf $dir
