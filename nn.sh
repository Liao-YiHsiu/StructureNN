#!/bin/bash

echo "$0 $@"  # Print the command line for logging

. parse_options.sh || exit 1;

files="train.lab test.lab train.ark test.ark train.lat test.lat"

if [ "$#" -ne 1 ]; then
   echo "Train Structure SVM with NN on a data set"
   echo "Usage: $0 <dir> "
   echo "eg. $0 data/simp"
   echo ""
   echo "dir-> $files"
   exit 1;
fi

dir=$1
log=$dir/data_nn_${C}.log
model=$dir/data_nn_${C}.model


   #check file existence.
   for file in $files;
   do
      [ -f $dir/$file ] || ( echo "File '$dir/$file' not found." && exit 1 );
   done

   #generate svm file
   [ -f $dir/data.out ] || con-svm ark:$dir/train.lab ark,s,cs:$dir/train.ark $dir/data.out  
   [ -f $dir/test.out ] || con-svm ark:$dir/test.lab ark,s,cs:$dir/test.ark $dir/test.out  

   echo "SVM with NN training start..................................."

   snnet/train.sh $dir/data.out ark:$dir/train.lat $model &> $log
   
   echo "SVM with NN testing start..................................."

   snnet/test.sh $dir/test.out ark:$dir/test.lat $model &>> $log


exit 0;
