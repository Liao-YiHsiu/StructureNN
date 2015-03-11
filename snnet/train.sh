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

timit_root=~/Research/timit

# Begin configuration.
config=

# training options
l1_penalty=0
l2_penalty=0
max_iters=200
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
learn_rate=0.004
momentum=0.9
minibatch_size=256
randomizer_size=32768
negative_num=100
GibbsIter=1000
num_inference=10
error_function="fer"
train_tool="snnet-train-shuff"
test_tool="snnet-gibbs"
dnn_depth=1
dnn_width=200
early_stop=1.0
train_opt=
# End configuration.

echo "$0 $@"  # Print the command line for logging

. parse_options.sh || exit 1;

if [ "$#" -ne 7 ]; then
   echo "Perform structure DNN training"
   echo "Usage: $0 <feat-rspecifier> <label-rspecifier> <lattice-rspecifier> \\"
   echo "          <cv-feat-rspecifier> <cv-label-rspecifier> <cv-lattice-rspecifier> <model_out>"
   echo "eg. $0 ark:1.ark ark:1.lab ark:1.lat ark:dev.ark ark:dev.lab ark:dev.lat model"
   echo ""
   exit 1;
fi

feat_data=$1
label_data=$2
lattice_data=$3

cv_feat_data=$4
cv_label_data=$5
cv_lattice_data=$6

init_path=
model_out=$7

[ ! -d $dir/nnet ] && mkdir $dir/nnet
[ ! -d $dir/log ] && mkdir $dir/log

#initialize model
max_state=$(copy-int-vector "$label_data" ark,t:-| cut -f 2- -d ' ' | tr " " "\n" | awk 'n < $0 {n=$0}END{print n}')
feat_dim=$(feat-to-dim "$feat_data" -)
SVM_dim=$(( (max_state + feat_dim) * max_state ))
mlp_init=$dir/nnet.init
mlp_proto=$dir/nnet.proto
$timit_root/utils/nnet/make_nnet_proto.py $SVM_dim 2 $dnn_depth $dnn_width > $mlp_proto || exit 1
nnet-initialize $mlp_proto $mlp_init || exit 1; 

init-score-path "$feat_data" ark:$dir/test.ark
init-score-path "$cv_feat_data" ark:$dir/cv.ark

mlp_best=$mlp_init
#TODO use lattice best path as init path to do gibbs
lattice-to-nbest "$lattice_data" ark:- | lattice-to-vec ark:- ark:- |split-path-score ark:- ark:/dev/null ark:$dir/train.lat
lattice-to-nbest "$cv_lattice_data" ark:- | lattice-to-vec ark:- ark:- |split-path-score ark:- ark:/dev/null ark:$dir/cv.lat

loss=0
halving=0
# start training
for iter in $(seq -w $max_iters); do
   mlp_next=$dir/nnet/nnet.${iter}

   if [ $((iter % num_inference)) -eq 0 ]; then
      # find negitive example
      log=$dir/log/iter${iter}.ptr.log; hostname>$log
      $test_tool --seed=$seed --GibbsIter=$GibbsIter --early-stop=$early_stop \
         "$feat_data" $mlp_best ark:$dir/test_tmp.ark \
         ${init_path:+ --init-path=ark:$dir/train.lat} \
         2>&1 | tee -a $log ; ( exit ${PIPESTATUS[0]} ) || exit 1;

      combine-score-path ark:$dir/test_tmp2.ark ark:$dir/test_tmp.ark ark:$dir/test.ark
      2>&1 | tee -a $log ; ( exit ${PIPESTATUS[0]} ) || exit 1;

      cp -f $dir/test_tmp2.ark $dir/test.ark
      seed=$((seed + 1))
   fi


# train
   log=$dir/log/iter${iter}.tr.log; hostname>$log

   $train_tool \
      --learn-rate=$learn_rate --momentum=$momentum --l1-penalty=$l1_penalty --l2-penalty=$l2_penalty \
      --minibatch-size=$minibatch_size --randomizer-size=$randomizer_size --randomize=true \
      --verbose=$verbose --binary=true --randomizer-seed=$seed \
      --negative-num=$negative_num --error-function=$error_function "$train_opt"\
      "$feat_data" "$label_data" ark:$dir/test.ark \
      $mlp_best $mlp_next \
      2>&1 | tee -a $log ; ( exit ${PIPESTATUS[0]} ) || exit 1;

      seed=$((seed + 1))
   loss_tr=$(cat $log | grep "AvgLoss:" | tail -n 1 | awk '{ print $4; }')
   echo -n "TRAIN AVG.LOSS $(printf "%.4f" $loss_tr), "

# CV
# find negitive example
#   log=$dir/log/iter${iter}.ptr.log; hostname>$log
#   $test_tool --seed=$seed --GibbsIter=$GibbsIter --early-stop=$early_stop\
#      "$cv_feat_data" $mlp_next ark:$dir/cv.ark \
#      2>&1 | tee -a $log ; ( exit ${PIPESTATUS[0]} ) || exit 1;
#
#   seed=$((seed + 1))

   log=$dir/log/iter${iter}.cv.log; hostname>$log
#   $train_tool \
#      --learn-rate=$learn_rate --momentum=$momentum --l1-penalty=$l1_penalty --l2-penalty=$l2_penalty \
#      --minibatch-size=$minibatch_size --randomizer-size=$randomizer_size --randomize=true \
#      --verbose=$verbose --binary=true --randomizer-seed=$seed\
#      --cross-validate=true \
#      --negative-num=0 --error-function=$error_function\
#      "$cv_feat_data" "$cv_label_data" ark:$dir/cv.ark \
#      $mlp_next \
#      2>&1 | tee -a $log ; ( exit ${PIPESTATUS[0]} ) || exit 1;
#      seed=$((seed + 1))
#
#
#   loss_new=$(cat $log | grep "AvgLoss:" | tail -n 1 | awk '{ print $4; }')
#   echo -n "CROSSVAL AVG.LOSS $(printf "%.4f" $loss_new), "

   $test_tool --seed=$seed --GibbsIter=$GibbsIter \
      ${init_path:+ --init-path=ark:$dir/cv.lat} \
      "$cv_feat_data" $mlp_next ark:$dir/cv.ark \
      2>&1 | tee -a $log ; ( exit ${PIPESTATUS[0]} ) || exit 1;

   seed=$((seed + 1))

   path-fer $cv_label_data "ark:split-path-score ark:$dir/cv.ark ark:/dev/null ark:- |" \
      2>&1 | tee -a $log ; ( exit ${PIPESTATUS[0]} ) || exit 1;

   loss_new=$(cat $log | grep 'Frame Error Rate' | tail -n 1 | awk '{ print $7; }')
   echo -n "CROSSVAL FER= $(printf "%.4f" $loss_new), "


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



