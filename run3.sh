#!/bin/bash
source path

dnn_depth=1
dnn_width=32
mux_width=32
test_lattice_N=10
lattice_N=50
train_opt=
momentum=0.9
learn_rate=0.000002
cpus=$(nproc)
acwt=0.16
lat_model=$timit/exp/dnn4_pretrain-dbn_dnn_smbr/final.mdl
feature_transform=
keep_lr_iters=100
dnn1_depth=4
dnn1_width=32
num_stream=5

sigma=
debug=

echo "$0 $@"  # Print the command line for logging
command_line="$0 $@"

. parse_options.sh || exit 1;

files="train.lab dev.lab test.lab train.ark dev.ark test.ark train.lat dev.lat test.lat nnet1"

if [ "$#" -ne 1 ]; then
   echo "Train Structure SVM with NN on a data set"
   echo "Usage: $0 <dir> "
   echo "eg. $0 data/simp"
   echo ""
   echo "dir-> $files"
   exit 1;
fi

train_tool="msnnet-train-listshuff \
   ${num_stream:+ --num-stream=$num_stream} \
   ${sigma:+ --sigma=$sigma}" 

dir=$1
paramId=${dnn_depth}_${dnn_width}_${mux_width}_${dnn1_depth}_${dnn1_width}_${lattice_N}_${test_lattice_N}_${learn_rate}_${acwt}_${keep_lr_iters}_${train_tool// /}

log=log/$dir/${paramId}.log
data=$dir/$paramId/data
model=$dir/$paramId/nnet

[ ! -d log/$dir ] && mkdir -p log/$dir
[ ! -d $dir/../lab ] && mkdir -p $dir/../lab

echo "$HOSTNAME `date`" \
2>&1 | tee $log ; ( exit ${PIPESTATUS[0]} ) || exit 1;

echo "$command_line" \
2>&1 | tee -a $log ; ( exit ${PIPESTATUS[0]} ) || exit 1;

stateMax=$(copy-int-vector "ark:$dir/train32.lab" ark,t:-| cut -f 2- -d ' ' | tr " " "\n" | awk 'n < $0 {n=$0}END{print n}')

   #check file existence.
   for file in $files;
   do
      [ -f $dir/$file ] || ( echo "File '$dir/$file' not found." && exit 1 );
   done

   [ -f $dir/transf.nnet ] && feature_transform=$dir/transf.nnet

   [ ! -d $dir/$paramId ] && mkdir -p $dir/$paramId

   [ -f $model ] || msnnet_train.sh --cpus $cpus\
      --dnn-depth $dnn_depth --dnn-width $dnn_width \
      --learn-rate $learn_rate --acwt $acwt \
      --momentum $momentum\
      --train-tool "$train_tool" \
      --test-lattice-N ${test_lattice_N} --lattice-N $lattice_N\
      ${mux_width:+ --mux-width $mux_width} ${debug:+ --debug $debug} \
      ${keep_lr_iters:+ --keep-lr-iters $keep_lr_iters} \
      ${train_opt:+ --train-opt "$train_opt"} \
      ${keep_lr_iters:+ --keep-lr-iters "$keep_lr_iters"} \
      ${feature_transform:+ --feature-transform "$feature_transform"} \
      ${dnn1_depth:+ --dnn1-depth $dnn1_depth} \
      ${dnn1_width:+ --dnn1-width $dnn1_width} \
      $dir $lat_model $model $stateMax \
      2>&1 | tee -a $log ; ( exit ${PIPESTATUS[0]} ) || exit 1;

   test_lattice_path=$dir/../lab/test.lab_${test_lattice_N}_${acwt}.gz
   lattice-to-nbest-path.sh --cpus $cpus --acoustic-scale $acwt --n $test_lattice_N \
      $lat_model ark:$dir/test.lat "$test_lattice_path" \
      2>&1 | tee -a $log ; ( exit ${PIPESTATUS[0]} ) || exit 1;

   msnnet-score \
      ${feature_transform:+ --feature-transform="$feature_transform"} \
      ark:$dir/test.ark \
      "ark:gunzip -c $test_lattice_path |" \
      $model "ark:| gzip -c > ${data}.tag.gz"\
      2>&1 | tee -a $log ; ( exit ${PIPESTATUS[0]} ) || exit 1;

   best-score-path "ark:gunzip -c ${data}.tag.gz |" ark:${data}.tag.1best
      2>&1 | tee -a $log ; ( exit ${PIPESTATUS[0]} ) || exit 1;

   calc.sh ark:$dir/test.lab ark:${data}.tag.1best \
      2>&1 | tee -a $log ; ( exit ${PIPESTATUS[0]} ) || exit 1;

echo "Finish Time: `date`" \
2>&1 | tee -a $log ; ( exit ${PIPESTATUS[0]} ) || exit 1;

exit 0;
