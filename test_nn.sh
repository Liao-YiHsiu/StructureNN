#!/bin/bash

echo "$0 $@"  # Print the command line for logging

. parse_options.sh || exit 1;

if [ "$#" -ne 2 ]; then
   echo "Do SNN Gibbs based on lattice data."
   echo "Usage: $0 <model> <dir>" 
   echo "eg. $0 data/nn_post/data_nn.model data/nn_post"
   echo ""
   exit 1;
fi

model=$1
dir=$2
tmp_dir=$(mktemp -d)

files="test.lat test.lab test.ark"
#check file existence.
for file in $files;
do
   if [ ! -f $dir/$file ]; then
      echo "File '$dir/$file' not found."
      exit 1;
   fi

done


   lattice-to-nbest ark:$dir/test.lat ark:- | lattice-to-vec ark:- ark:- |split-path-score ark:- ark:/dev/null ark:$tmp_dir/test.best

   snnet-gibbs --init-path=ark:$tmp_dir/test.best ark:$dir/test.ark $model ark,t:$tmp_dir/test.tag || exit 1;
   path-fer ark:$dir/test.lab "ark:split-path-score ark:${tmp_dir}/test.tag ark:/dev/null ark:- |" || exit 1;

   compute-wer "ark:trim-path ark:$dir/test.lab ark:- |" "ark:split-path-score ark:$tmp_dir/test.tag ark:/dev/null ark:- | trim-path ark:- ark:- |" || exit 1;

   echo "Calculating Error rate.(39)"
   path-fer "ark:trans.sh ark:$dir/test.lab ark:- |" "ark:split-path-score ark:$tmp_dir/test.tag ark:/dev/null ark:- | trans.sh ark:- ark:- |" || exit 1;

   compute-wer "ark:trim-path ark:$dir/test.lab ark:- | trans.sh ark:- ark:- |" "ark:split-path-score ark:$tmp_dir/test.tag ark:/dev/null ark:- | trim-path ark:- ark:- | trans.sh ark:- ark:- |" || exit 1;

