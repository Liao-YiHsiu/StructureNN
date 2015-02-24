#!/bin/bash
GibbsIter=1000

echo "$0 $@"  # Print the command line for logging

. parse_options.sh || exit 1;

files="train.lab dev.lab test.lab train.ark dev.ark test.ark train.lat dev.lat test.lat"

if [ "$#" -ne 1 ]; then
   echo "Train Structure SVM with NN on a data set"
   echo "Usage: $0 <dir> "
   echo "eg. $0 data/simp"
   echo ""
   echo "dir-> $files"
   exit 1;
fi

dir=$1
log=$dir/data_nn.log
model=$dir/data_nn.model


   #check file existence.
   for file in $files;
   do
      [ -f $dir/$file ] || ( echo "File '$dir/$file' not found." && exit 1 );
   done

   echo "SVM with NN training start..................................."

   snnet/train.sh --GibbsIter $GibbsIter \
      ark:$dir/train.ark ark:$dir/train.lab ark:$dir/train.lat \
      ark:$dir/dev.ark   ark:$dir/dev.lab   ark:$dir/dev.lat $model \
      2>&1 | tee $log ; ( exit ${PIPESTATUS[0]} ) || exit 1;
   
   echo "SVM with NN testing start..................................."

   snnet-gibbs ark:$dir/test.ark $model ark,t:$dir/test.tags \
      2>&1 | tee -a $log ; ( exit ${PIPESTATUS[0]} ) || exit 1;


exit 0;
