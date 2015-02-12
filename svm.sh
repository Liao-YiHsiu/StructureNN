#!/bin/bash

C=1000

echo "$0 $@"  # Print the command line for logging

. parse_options.sh || exit 1;

if [ "$#" -ne 1 ]; then
   echo "Train Structure SVM on a data set"
   echo "Usage: $0 <dir> "
   echo "eg. $0 data/simp"
   echo ""
   echo "dir-> train.lab test.lab train.ark test.ark"
   exit 1;
fi

dir=$1
log=$dir/data_${C}.log
model=$dir/data_${C}.model

   #check file existence.
   files="train.lab test.lab train.ark test.ark"
   for file in $files;
   do
      [ -f $dir/$file ] || ( echo "File '$dir/$file' not found." && exit 1 );
   done

   
   #generate svm file
   [ -f $dir/data.out ] || con-svm ark:$dir/train.lab ark,s,cs:$dir/train.ark $dir/data.out  
   [ -f $dir/test.out ] || con-svm ark:$dir/test.lab ark,s,cs:$dir/test.ark $dir/test.out  

   echo "SVM training start..................................."
   [ -f $model ] || svm_hmm/svm_hmm_learn -c $C -e 0.5 $dir/data.out $model &> $log 
   
   echo "SVM testing start..................................."
   svm_hmm/svm_hmm_classify $dir/test.out $model $dir/test.tags &>> $log 
   svm_hmm/svm_hmm_classify $dir/data.out $model $dir/data.tags &>> $log 

exit 0;

