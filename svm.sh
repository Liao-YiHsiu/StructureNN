#!/bin/bash

C=1000

echo "$0 $@"  # Print the command line for logging

. parse_options.sh || exit 1;

files="train.lab dev.lab test.lab train.ark dev.ark test.ark"

if [ "$#" -ne 1 ]; then
   echo "Train Structure SVM on a data set"
   echo "Usage: $0 <dir> "
   echo "eg. $0 data/simp"
   echo ""
   echo "dir-> $files"
   exit 1;
fi

dir=$1
log=$dir/data_${C}.log
model=$dir/data_${C}.model

   #check file existence.
   for file in $files;
   do
      [ -f $dir/$file ] || ( echo "File '$dir/$file' not found." && exit 1 );
   done

   
   #generate svm file
   [ -f $dir/data.out ] || con-svm ark:$dir/train.lab ark:$dir/train.ark $dir/data.out  
   [ -f $dir/dev.out ]  || con-svm ark:$dir/dev.lab   ark:$dir/dev.ark $dir/dev.out  
   [ -f $dir/test.out ] || con-svm ark:$dir/test.lab ark:$dir/test.ark $dir/test.out  

   echo "SVM training start..................................."
   [ -f $model ] || svm_hmm/svm_hmm_learn -c $C -e 0.5 $dir/data.out $model &> $log 
   
   echo "SVM testing start..................................."
   svm_hmm/svm_hmm_classify $dir/test.out $model $dir/test.tags &>> $log 
   svm_hmm/svm_hmm_classify $dir/dev.out $model $dir/dev.tags &>> $log 
   svm_hmm/svm_hmm_classify $dir/data.out $model $dir/data.tags &>> $log 

exit 0;

