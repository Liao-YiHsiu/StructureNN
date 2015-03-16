#!/bin/bash
error_function="per"
dnn_depth=1
dnn_width=200
lattice_N=1000
train_opt=
keep_lr_iters=
cpus=10

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
log=$dir/data_nn.log_${lattice_N}_${dnn_depth}_${dnn_width}_${train_opt}_${keep_lr_iters}
model=$dir/data_nn.model_${lattice_N}_${dnn_depth}_${dnn_width}_${train_opt}_${keep_lr_iters}

echo "$0 $@" \
2>&1 | tee $log ; ( exit ${PIPESTATUS[0]} ) || exit 1;

   #check file existence.
   for file in $files;
   do
      [ -f $dir/$file ] || ( echo "File '$dir/$file' not found." && exit 1 );
   done

   echo "SVM with NN training start..................................."\
      2>&1 | tee $log ; ( exit ${PIPESTATUS[0]} ) || exit 1;

   [ -f $model ] || snnet/train.sh --error-function $error_function --cpus $cpus\
      --dnn-depth $dnn_depth --dnn-width $dnn_width --lattice-N $lattice_N\
      ${train_opt:+ --train-opt "$train_opt"} \
      ${keep_lr_iters:+ --keep-lr-iters "$keep_lr_iters"} \
      ark:$dir/train.ark ark:$dir/train.lab ark:$dir/train.lat \
      ark:$dir/dev.ark   ark:$dir/dev.lab   ark:$dir/dev.lat $model \
      2>&1 | tee $log ; ( exit ${PIPESTATUS[0]} ) || exit 1;

   lattice-to-nbest-cpus.sh --cpus $cpus --n $((lattice_N * 2)) ark:$dir/test.lat ark:- | lattice-to-vec ark:- ark:$dir/test_best.lat \
      2>&1 | tee $log ; ( exit ${PIPESTATUS[0]} ) || exit 1;

   snnet-best ark:$dir/test.ark ark:$dir/test_best.lat $model ark,t:${model}.tag\
      2>&1 | tee -a $log ; ( exit ${PIPESTATUS[0]} ) || exit 1;

   echo "Calculating Error rate." \
      2>&1 | tee -a $log ; ( exit ${PIPESTATUS[0]} ) || exit 1;

   path-fer ark:$dir/test.lab "ark:split-path-score ark:${model}.tag ark:/dev/null ark:- |" \
      2>&1 | tee -a $log ; ( exit ${PIPESTATUS[0]} ) || exit 1;

   compute-wer "ark:trim-path ark:$dir/test.lab ark:- |" "ark:split-path-score ark:${model}.tag ark:/dev/null ark:- | trim-path ark:- ark:- |" \
      2>&1 | tee -a $log ; ( exit ${PIPESTATUS[0]} ) || exit 1;

   echo "Calculating Error rate.(39)" \
      2>&1 | tee -a $log ; ( exit ${PIPESTATUS[0]} ) || exit 1;

   path-fer "ark:trans.sh ark:$dir/test.lab ark:- |" "ark:split-path-score ark:${model}.tag ark:/dev/null ark:- | trans.sh ark:- ark:- |" \
      2>&1 | tee -a $log ; ( exit ${PIPESTATUS[0]} ) || exit 1;

   compute-wer "ark:trim-path ark:$dir/test.lab ark:- | trans.sh ark:- ark:- |" "ark:split-path-score ark:${model}.tag ark:/dev/null ark:- | trim-path ark:- ark:- | trans.sh ark:- ark:- |" \
      2>&1 | tee -a $log ; ( exit ${PIPESTATUS[0]} ) || exit 1;

exit 0;
