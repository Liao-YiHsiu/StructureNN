#!/bin/bash

function tobyte {
   num=$1
   case $num in
      *[Kk]*) num=${num%[Kk]*}000 ;;
      *[Mm]*) num=${num%[Mm]*}000000 ;;
      *[Gg]*) num=${num%[Gg]*}000000000 ;;
   esac
   echo $num
}

# perform bottleneck feature extraction.
# usage: ./bottleneck.sh <rspecifier> <bottleneck-number> <model>

timit_root=../../timit

. path.sh
. cmd.sh

# Begin configuration.
config=

# training options
l1_penalty=0
l2_penalty=0
max_iters=100
min_iters=
keep_lr_iters=1
start_halving_impr=0.001
end_halving_impr=0.0001
halving_factor=0.5

verbose=1

frame_weights=
dir=$(mktemp -d)
bn_dim=48
mlp_proto=
seed=777
learn_rate=0.00001
momentum=0.9
minibatch_size=256
randomizer_seed=777
randomizer_size=32768
cv_percent=10
cut_size="1G"
train_tool="nnet-train-frmshuff"
# End configuration.

echo "$0 $@"  # Print the command line for logging

. parse_options.sh || exit 1;

if [ "$#" -ne 3 ]; then
   echo "Perform bottleneck feature extraction on a probability input."
   echo "Usage: $0 <rspecifier> <cv-rspecifier> <model-out>"
   echo "eg. $0 scp:feat.scp scp:cv_feat.scp model.nnet"
   echo ""
   exit 1;
fi


data=$1
cv_data=$2
model=$3

#generate scp
copy-feats "$cv_data" ark,scp:$dir/feats_cv.ark,$dir/feats_cv.scp || exit 1
copy-feats "$data" ark,scp:$dir/feats_tr.ark,$dir/feats_tr.scp || exit 1

cut_size=$(tobyte $cut_size)
data_size=$(du -sb $dir/feats_tr.ark | cut -f 1)
data_num=$(( data_size / cut_size ))

echo "data cut into $data_num chunks $data_size $cut_size"

#generate NN proto
if [ -z "$mlp_proto" ]; then 
   feat_dim=$(feat-to-dim --print-args=false scp:$dir/feats_cv.scp -)
   mlp_proto=$dir/nnet.proto
   $timit_root/utils/nnet/make_nnet_proto.py $feat_dim $feat_dim 1 $bn_dim > $mlp_proto || exit 1
fi

# INITIALIZE NNET
mlp_init=$dir/nnet.init
nnet-initialize $mlp_proto $mlp_init || exit 1; 

cp  ${dir}/feats_cv.scp $dir/cv.scp || exit 1;
feats_cv="scp:$dir/cv.scp"
labels_cv="ark:feat-to-post scp:$dir/cv.scp ark:- |"

[ ! -d $dir/nnet ] && mkdir $dir/nnet
[ ! -d $dir/log ] && mkdir $dir/log

mlp_best=$mlp_init

loss=0
halving=0
# start training
for iter in $(seq -w $max_iters); do

   # do training.
   # shuffle training list
   cat ${dir}/feats_tr.scp | $timit_root/utils/shuffle_list.pl --srand $seed > $dir/train.scp || exit 1;
   seed=$(( seed + 1 ))

   mlp_pre=$mlp_best

   # cut the data into several peices
   for (( num=0 ; num<data_num ; num++ ))
   do
      mlp_next=$dir/nnet/nnet.${iter}.${num}

      $timit_root/utils/split_scp.pl -j $data_num $num $dir/train.scp $dir/tmp.scp
      feats_tr="scp:$dir/tmp.scp"
      labels_tr="ark:feat-to-post scp:$dir/tmp.scp ark:- |"

      # training
      log=$dir/log/iter${iter}.${num}.tr.log; hostname>$log
      $train_tool \
       --learn-rate=$learn_rate --momentum=$momentum --l1-penalty=$l1_penalty --l2-penalty=$l2_penalty \
       --minibatch-size=$minibatch_size --randomizer-size=$randomizer_size --randomize=true \
       --verbose=$verbose --binary=true --randomizer-seed=$randomizer_seed \
       ${feature_transform:+ --feature-transform=$feature_transform} \
       ${frame_weights:+ "--frame-weights=$frame_weights"} \
       "$feats_tr" "$labels_tr" $mlp_pre $mlp_next \
       2>> $log || exit 1; 
      randomizer_seed=$((randomizer_seed + 1))

      tr_loss=$(cat $log | grep "AvgLoss:" | tail -n 1 | awk '{ print $4; }')
      echo -n "TRAIN AVG.LOSS $(printf "%.4f" $tr_loss), (lrate$(printf "%.6g" $learn_rate)), "

      mlp_pre=$mlp_next
   done

   # cross-validation
   log=$dir/log/iter${iter}.cv.log; hostname>$log
   $train_tool --cross-validate=true \
    --minibatch-size=$minibatch_size --randomizer-size=$randomizer_size --randomize=false \
    --verbose=$verbose \
    ${feature_transform:+ --feature-transform=$feature_transform} \
    ${frame_weights:+ "--frame-weights=$frame_weights"} \
    "$feats_cv" "$labels_cv" $mlp_next \
    2>>$log || exit 1;
   
   loss_new=$(cat $log | grep "AvgLoss:" | tail -n 1 | awk '{ print $4; }')
   echo -n "CROSSVAL AVG.LOSS $(printf "%.4f" $loss_new), "


   # accept or reject new parameters (based on objective function)
   loss_prev=$loss
   if [ 1 == $(bc <<< "$loss_new < $loss") -o $iter -le $keep_lr_iters ]; then
      loss=$loss_new
      mlp_best=$dir/nnet/nnet.${iter}_learnrate${learn_rate}_tr$(printf "%.4f" $tr_loss)_cv$(printf "%.4f" $loss_new)
      [ $iter -le $keep_lr_iters ] && mlp_best=${mlp_best}_keep-lr-iters-$keep_lr_iters
      mv $mlp_next $mlp_best
      echo "nnet accepted ($(basename $mlp_best))"
   else
     mlp_reject=$dir/nnet/nnet.${iter}_learnrate${learn_rate}_tr$(printf "%.4f" $tr_loss)_cv$(printf "%.4f" $loss_new)_rejected
     mv $mlp_next $mlp_reject
     echo "nnet rejected ($(basename $mlp_reject))"
   fi

   # no learn-rate halving yet, if keep_lr_iters set accordingly
   [ $iter -le $keep_lr_iters ] && continue 

   # stopping criterion
   rel_impr=$(bc <<< "scale=10; ($loss_prev-$loss)/$loss_prev")
   if [ 1 == $halving -a 1 == $(bc <<< "$rel_impr < $end_halving_impr") ]; then
     if [[ "$min_iters" != "" ]]; then
       if [ $min_iters -gt $iter ]; then
         echo we were supposed to finish, but we continue as min_iters : $min_iters
         continue
       fi
     fi
     echo finished, too small rel. improvement $rel_impr
     break
   fi

   # start annealing when improvement is low
   if [ 1 == $(bc <<< "$rel_impr < $start_halving_impr") ]; then
     halving=1
   fi
   
   # do annealing
   if [ 1 == $halving ]; then
     learn_rate=$(awk "BEGIN{print($learn_rate*$halving_factor)}")
   fi
   
done

echo "train success"

nnet-pop --num=2 $mlp_best $model
exit 0;


