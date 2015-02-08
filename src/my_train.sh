#!/bin/bash


. path.sh
. cmd.sh

# Begin configuration.
config=
train_tool="nnet-train-frmshuff"
train_seq="nnet-train-perutt"
cut_size="4G"
max_peice=100

# data processing
train_all="learn_rate momentum l1_penalty l2_penalty minibatch_size randomizer_size randomizer_seed randomize verbose feature_transform frame_weights cross_validate objective_function binary dropout_retention use_gpu"
train_opts=

# End configuration.

echo "$0 $@"  # Print the command line for logging

. parse_options.sh || exit 1;

for opt in $train_all; do 
   if [ ! -z "$opt" ]; then
      train_opts="$train_opts --$opt=$(eval \$$opt)"
   fi
done

if [ "$#" -ne 3 ] && [ "$#" -ne 4 ]; then
   echo "Perform NN train but cut data into slices randomly and feed data into $train_tool"
   echo "Usage: $0 [options] <feature-rspecifier> <targets-rspecifier> <model-in> [<model-out>]"
   echo "eg. $0 --binary=false scp:feats.scp ark:posterior.ark nnet.init nnet.iter1"
   echo ""
   exit 1;
fi

feats=$1
labels=$2
model_in=$3
model_out=$4

# no randomize no need to cut.
# can be improved if we use nnet-train-perutt
if [ "$randomize" == "false"]; then
#$train_tool $train_opts $feats $labels $model_in $model_out
   $train_seq $train_opts $feats $labels $model_in $model_out
   exit $?;
fi

tmp_dir=$(mktemp -d)

# cut data into several peices

rand_num=$RANDOM
rand-feats --rand-seed=$rand_num --cut-size=$cut_size $feats $tmp_dir/feats || exit 1;
rand-feats --rand-seed=$rand_num --cut-size=$cut_size $labels $tmp_dir/labels || exit 1;

# train each peices one at a time.
pre="init"
cp $model_in $tmp_dir/model.$pre || exit 1;

for index in $(seq -w $max_peice); do
do
   if [ -f "$tmp_dir/feats${index}.scp" ] && [ -f "$tmp_dir/labels${index}.scp" ] ; then
      $train_tool $train_opts $tmp_dir/feats${index}.scp $tmp_dir/labels${index}.scp \
         $tmp_dir/model.$pre $tmp_dir/model.$index || exit 1;
   else
      cp $tmp_dir/model.$pre $model_out || exit 1;
      break;
   fi
   pre=$index
done


