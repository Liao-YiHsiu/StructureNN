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

timit_root=../timit

# Begin configuration.
config=

# training options
l1_penalty=0
l2_penalty=0
max_iters=100
min_iters=
keep_lr_iters=1
start_halving_impr=0.005
end_halving_impr=0.0005
halving_factor=0.5

verbose=1

frame_weights=
dir=$(mktemp -d)
mlp_proto=
seed=777
learn_rate=0.00001
momentum=0.9
minibatch_size=1024
randomizer_size=10240
cv_percent=10
n_lattice=300
rand_lattice="true"
train_tool="snnet-train-shuff"
# End configuration.

echo "$0 $@"  # Print the command line for logging

. parse_options.sh || exit 1;

if [ "$#" -ne 3 ]; then
   echo "Perform structure DNN training"
   echo "Usage: $0 <Structure-SVM-In> <lattice-rspecifier> <model_out>"
   echo "eg. $0 data.out ark:1.lat model"
   echo ""
   exit 1;
fi

svm_data=$1
lattice_data=$2
model_out=$3

[ ! -d $dir/nnet ] && mkdir $dir/nnet
[ ! -d $dir/log ] && mkdir $dir/log

#initialize model
feat_dim=$(svm-info $svm_data)
mlp_init=$dir/nnet.init
mlp_proto=$dir/nnet.proto
$timit_root/utils/nnet/make_nnet_proto.py $feat_dim 2 3 100 > $mlp_proto || exit 1
nnet-initialize $mlp_proto $mlp_init || exit 1; 

mlp_best=$model_init

loss=0
halving=0
# start training
for iter in $(seq -w $max_iters); do
   mlp_next=$dir/nnet/nnet.${iter}

   $train_tool \
      --learn-rate=$learn_rate --momentum=$momentum --l1-penalty=$l1_penalty --l2-penalty=$l2_penalty \
      --minibatch-size=$minibatch_size --randomizer-size=$randomizer_size --randomize=true \
      --verbose=$verbose --binary=true --randomizer-seed=$seed --cv-percent=$cv_percent\
      ${feature_transform:+ --feature-transform=$feature_transform} \
      ${frame_weights:+ "--frame-weights=$frame_weights"} \
      $svm_data "ark:lattice-to-nbest --n=$n_lattice --random=$rand_lattice ark:$lattice_data ark:- | lattice-to-vec ark:- ark:- |" \
      $mlp_best $mlp_next \
      2>> $log || exit 1; 
      seed=$((seed + 1))


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

cp $mlp_best $model_out

exit 0;



