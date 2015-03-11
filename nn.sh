#!/bin/bash
GibbsIter=1000
error_function="per"
dnn_depth=1
dnn_width=200
early_stop=1.0
num_inference=1
train_opt=
init_path=

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
log=$dir/data_nn.log_${GibbsIter}_${dnn_depth}_${dnn_width}_${num_inference}_${train_opt}_${init_path}
model=$dir/data_nn.model_${GibbsIter}_${dnn_depth}_${dnn_width}_${num_inference}_${train_opt}_${init_path}


   #check file existence.
   for file in $files;
   do
      [ -f $dir/$file ] || ( echo "File '$dir/$file' not found." && exit 1 );
   done

   echo "SVM with NN training start..................................."

   [ -f $model ] || snnet/train.sh --GibbsIter $GibbsIter --error-function $error_function \
      --dnn-depth $dnn_depth --dnn-width $dnn_width --early-stop $early_stop\
      --num-inference $num_inference \
      ${train_opt:+ --train-opt "$train_opt"} \
      ${init_path:+ --init-path "$init_path"} \
      ark:$dir/train.ark ark:$dir/train.lab ark:$dir/train.lat \
      ark:$dir/dev.ark   ark:$dir/dev.lab   ark:$dir/dev.lat $model \
      2>&1 | tee $log ; ( exit ${PIPESTATUS[0]} ) || exit 1;
   
   echo "SVM with NN testing start..................................."

   snnet-gibbs ark:$dir/test.ark $model ark,t:$dir/test_nn.tags \
      2>&1 | tee -a $log ; ( exit ${PIPESTATUS[0]} ) || exit 1;

   echo "Calculating Error rate."

   path-fer ark:$dir/test.lab "ark:split-path-score ark:$dir/test_nn.tags ark:/dev/null ark:- |" \
      2>&1 | tee -a $log ; ( exit ${PIPESTATUS[0]} ) || exit 1;

   compute-wer "ark:trim-path ark:$dir/test.lab ark:- |" "ark:split-path-score ark:$dir/test_nn.tags ark:/dev/null ark:- | trim-path ark:- ark:- |" \
      2>&1 | tee -a $log ; ( exit ${PIPESTATUS[0]} ) || exit 1;

   echo "Calculating Error rate.(39)"
   path-fer "ark:trans.sh ark:$dir/test.lab ark:- |" "ark:split-path-score ark:$dir/test_nn.tags ark:/dev/null ark:- | trans.sh ark:- ark:- |" \
      2>&1 | tee -a $log ; ( exit ${PIPESTATUS[0]} ) || exit 1;

   compute-wer "ark:trim-path ark:$dir/test.lab ark:- | trans.sh ark:- ark:- |" "ark:split-path-score ark:$dir/test_nn.tags ark:/dev/null ark:- | trim-path ark:- ark:- | trans.sh ark:- ark:- |" \
      2>&1 | tee -a $log ; ( exit ${PIPESTATUS[0]} ) || exit 1;


exit 0;
