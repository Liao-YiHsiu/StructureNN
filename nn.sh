#!/bin/bash
error_function="per"
dnn_depth=1
dnn_width=200
lattice_N=1000
train_opt=
learn_rate=0.0005
cpus=10
acwt=0.16
lat_model=timit/exp/dnn4_pretrain-dbn_dnn_smbr/final.mdl

echo "$0 $@"  # Print the command line for logging
command_line="$0 $@"

source path
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
log=$dir/data_nn.log_${lattice_N}_${dnn_depth}_${dnn_width}__${learn_rate}_${acwt}
model=$dir/data_nn.model_${lattice_N}_${dnn_depth}_${dnn_width}__${learn_rate}_${acwt}

lattice_N_times=$((lattice_N))

echo $command_line \
2>&1 | tee $log ; ( exit ${PIPESTATUS[0]} ) || exit 1;

   #check file existence.
   for file in $files;
   do
      [ -f $dir/$file ] || ( echo "File '$dir/$file' not found." && exit 1 );
   done

   echo "SVM with NN training start..................................."\
      2>&1 | tee -a $log ; ( exit ${PIPESTATUS[0]} ) || exit 1;

   [ -f $model ] || snnet/train.sh --error-function $error_function --cpus $cpus\
      --dnn-depth $dnn_depth --dnn-width $dnn_width --lattice-N $lattice_N\
      --learn-rate $learn_rate --acwt $acwt \
      ${train_opt:+ --train-opt "$train_opt"} \
      ${keep_lr_iters:+ --keep-lr-iters "$keep_lr_iters"} \
      $dir $lat_model $model \
      2>&1 | tee -a $log ; ( exit ${PIPESTATUS[0]} ) || exit 1;

   test_lattice_path=$dir/test.lab_${lattice_N_times}_${acwt}.gz
   [ -f $test_lattice_path ] || lattice-to-nbest-path.sh --cpus $cpus --acoustic-scale $acwt  --n $lattice_N_times $lat_model ark:$dir/test.lat "ark:| gzip -c > $test_lattice_path" \
      2>&1 | tee -a $log ; ( exit ${PIPESTATUS[0]} ) || exit 1;

   snnet-score ark:$dir/test.ark "ark:gunzip -c $test_lattice_path |" $model ark:${model}.tag\
      2>&1 | tee -a $log ; ( exit ${PIPESTATUS[0]} ) || exit 1;

   best-score-path ark:${model}.tag ark:${model}.tag.1best
      2>&1 | tee -a $log ; ( exit ${PIPESTATUS[0]} ) || exit 1;

   calc.sh ark:$dir/test.lab ark:${model}.tag.1best \
      2>&1 | tee -a $log ; ( exit ${PIPESTATUS[0]} ) || exit 1;

#   log=$dir/data_nn.log_${lattice_N}_${dnn_depth}_${dnn_width}_${train_opt}_${keep_lr_iters}_${acwt}_sample
#   snnet-gibbs --init-path="ark:split-path-score ark:${model}.tag ark:/dev/null ark:- |" ark:$dir/test.ark $model ark,t:${model}.tag.sample\
#      2>&1 | tee -a $log ; ( exit ${PIPESTATUS[0]} ) || exit 1;
#
#   calc.sh ark:$dir/test.lab ark:${model}.tag.sample \
#      2>&1 | tee -a $log ; ( exit ${PIPESTATUS[0]} ) || exit 1;
exit 0;
