#!/bin/bash -ex

DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
source $DIR/../path
map_file=$DIR/mapping.txt

config=
help=
print_args=
verbose=

acoustic_scale=
n=
random=
srand=
cpus=$(( (`nproc` + 1)/3 ))
zip_file="true"

dir=$(mktemp -d)

echo "$0 $@" >&2  # Print the command line for logging

. parse_options.sh || exit 1;

if [ "$#" -ne 3 ]; then
   echo "a wrapper script for lattice-to-nbest and lattice-to-vec for mulitple cpu computation."
   echo "Usage: $0 [options] <model> <lattice-rspecifier> <score-path-wspecifier>"
   echo "e.g.: $0 --cpus 2 --acoustic-scale 0.1 --n 10 final.mdl ark:1.lats nbest.ark"
   echo ""
   exit 1;
fi

model=$1
lattice_r=$2
score_path_w=$3

if [ $cpus == 1 ]; then
   echo "Not for single cpu"
   exit 1;
fi

[ -e tmp ] && [ ! -d tmp ] && rm -rf tmp

[ ! -d tmp ] && mkdir tmp

lock_file=tmp/$(basename $score_path_w)

(
flock -w -1 9

# file already exists
if [ -f $score_path_w ]; then
   exit 0;
fi

if [ "$zip_file" == "true" ]; then
   score_path_w="ark:| gzip -c > $score_path_w"
else
   score_path_w="ark:$score_path_w"
fi

lattice-copy "$lattice_r" ark,scp:$dir/tmp.ark,$dir/tmp.scp || exit 1;

for i in $(seq 0 $((cpus - 1)) )
do
   file_in=$dir/in_${i}.scp
   file_out_scp=$dir/out_${i}.scp
   file_out_ark=$dir/out_${i}.ark

   $timit/utils/split_scp.pl -j $cpus $i $dir/tmp.scp $file_in || exit 1;
   cat >> $dir/parallel << EOF
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
   tee >(nbest-to-linear ark:- ark:/dev/null ark:/dev/null ark:${file_out_ark}.lm ark:${file_out_ark}.am) |\
   lattice-to-post ark:- ark:- |\
   post-to-phone-post "$model" ark:- ark:- | \
   post-to-vec --map-file=$map_file ark:- ark,scp:$file_out_ark,$file_out_scp 
EOF

done

   cat $dir/parallel | xargs -I CMD --max-procs=$cpus bash -c CMD || exit 1

file_out=$dir/out.scp
#combine jobs
for i in $(seq 0 $((cpus - 1)) )
do
   file_out_scp=$dir/out_${i}.scp
   file_out_ark=$dir/out_${i}.ark
   cat $file_out_scp >> $file_out
   cat ${file_out_ark}.lm >> ${file_out}.lm
   cat ${file_out_ark}.am >> ${file_out}.am
done

weight-basefloat ark:${file_out}.lm_am 1.0 ark:${file_out}.lm $acoustic_scale ark:${file_out}.am  || exit 1;

vec-to-score-path ${n:+ --n=$n} --score-rspecifier="ark:${file_out}.lm_am" scp:$file_out "$score_path_w" || exit 1;

) 9>$lock_file
rm -rf $lock_file
rm -rf $dir
