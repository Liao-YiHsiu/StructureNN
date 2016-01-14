#!/bin/bash -e
source path

test_lattice_N=10
lattice_N=20
train_opt=
momentum=0.9
learn_rate=0.0000001
cpus=$(nproc)
acwt=0.16
lat_model=$timit/exp/dnn4_pretrain-dbn_dnn_smbr/final.mdl
feature_transform=
keep_lr_iters=300
num_stream=4
seed=777
tmpdir=$(mktemp -d)

negative_num=
debug=

echo "$0 $@"  # Print the command line for logging
command_line="$0 $@"

. parse_options.sh || exit 1;

if [ "$debug" == "true" ]; then
   trap 'kill $(jobs -p) || true ' EXIT
else
   trap 'rm -rf $tmpdir; kill $(jobs -p) || true ' EXIT
fi

files="train.lab dev.lab test.lab train.ark dev.ark test.ark train.lat dev.lat test.lat"

if [ "$#" -ne 3 ]; then
   echo "Train Structure SVM with NN on a data set"
   echo "Usage: $0 <nnet_proto> <loss-func> <dir> "
   echo "eg. $0 nnet.proto loss.conf data/raw_feature"
   echo ""
   echo "dir-> $files"
   exit 1;
fi

train_tool="mynnet-train-utt \
   ${num_stream:+ --num-stream=$num_stream} "

nnet_proto=$1
loss_func=$2
dir=$3

nnet_init=$tmpdir/nnet.init

mynnet-init --seed=$seed $nnet_proto $nnet_init
sha=`(cat $nnet_init; cat $loss_func)|sha1sum| cut -b 1-6`

paramId=${sha}_${lattice_N}_${test_lattice_N}_${learn_rate}_${acwt}_${keep_lr_iters}_${train_tool// /}

log=log/$dir/${paramId}.log
data=$dir/$paramId/data
nnet=$dir/$paramId/nnet

[ ! -d log/$dir ] && mkdir -p log/$dir
[ ! -d $dir/../lab ] && mkdir -p $dir/../lab

echo "$HOSTNAME `date`" \
2>&1 | tee $log ; ( exit ${PIPESTATUS[0]} ) || exit 1;

echo "$command_line" \
2>&1 | tee -a $log ; ( exit ${PIPESTATUS[0]} ) || exit 1;

   #check file existence.
   for file in $files;
   do
      [ -f $dir/$file ] || ( echo "File '$dir/$file' not found." && exit 1 );
   done

   [ -f $dir/transf.nnet ] && feature_transform=$dir/transf.nnet

   [ ! -d $dir/$paramId ] && mkdir -p $dir/$paramId

   [ -f $nnet ] || mynnet_train.sh --cpus $cpus\
      --learn-rate $learn_rate --acwt $acwt \
      --momentum $momentum\
      --train-tool "$train_tool" \
      --test-lattice-N ${test_lattice_N} --lattice-N $lattice_N\
      ${debug:+ --debug $debug} ${seed:+ --seed $seed} \
      ${keep_lr_iters:+ --keep-lr-iters $keep_lr_iters} \
      ${train_opt:+ --train-opt "$train_opt"} \
      ${keep_lr_iters:+ --keep-lr-iters "$keep_lr_iters"} \
      ${feature_transform:+ --feature-transform "$feature_transform"} \
      $dir $lat_model $loss_func $nnet_init $nnet \
      2>&1 | tee -a $log ; ( exit ${PIPESTATUS[0]} ) || exit 1;

   test_lattice_path=$dir/../lab/test.lab_${test_lattice_N}_${acwt}.gz
   lattice-to-nbest-path.sh --cpus $cpus --acoustic-scale $acwt --n $test_lattice_N \
      $lat_model ark:$dir/test.lat "$test_lattice_path" \
      2>&1 | tee -a $log ; ( exit ${PIPESTATUS[0]} ) || exit 1;

   mynnet-score \
      ${feature_transform:+ --feature-transform="$feature_transform"} \
      ark:$dir/test.ark \
      "ark:gunzip -c $test_lattice_path |" \
      $nnet "ark:| gzip -c > ${data}.tag.gz"\
      2>&1 | tee -a $log ; ( exit ${PIPESTATUS[0]} ) || exit 1;

   best-score-path "ark:gunzip -c ${data}.tag.gz |" ark:${data}.tag.1best
      2>&1 | tee -a $log ; ( exit ${PIPESTATUS[0]} ) || exit 1;

   calc.sh ark:$dir/test.lab ark:${data}.tag.1best \
      2>&1 | tee -a $log ; ( exit ${PIPESTATUS[0]} ) || exit 1;

   mynnet-score \
      ${feature_transform:+ --feature-transform="$feature_transform"} \
      ark:$dir/test.ark \
      "ark:gunzip -c $test_lattice_path |" \
      ${nnet}.best "ark:| gzip -c > ${data}.best.tag.gz"\
      2>&1 | tee -a $log ; ( exit ${PIPESTATUS[0]} ) || exit 1;

   best-score-path "ark:gunzip -c ${data}.best.tag.gz |" ark:${data}.best.tag.1best
      2>&1 | tee -a $log ; ( exit ${PIPESTATUS[0]} ) || exit 1;

   calc.sh ark:$dir/test.lab ark:${data}.best.tag.1best \
      2>&1 | tee -a $log ; ( exit ${PIPESTATUS[0]} ) || exit 1;

echo "Finish Time: `date`" \
2>&1 | tee -a $log ; ( exit ${PIPESTATUS[0]} ) || exit 1;

exit 0;
