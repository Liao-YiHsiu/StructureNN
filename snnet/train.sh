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
DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
source $DIR/../path

# Begin configuration.
config=

# training options
l1_penalty=0
l2_penalty=0
max_iters=200
min_iters=
keep_lr_iters=1
start_halving_impr=0.005
end_halving_impr=0.00005
halving_factor=0.5

verbose=1

frame_weights=
tmpdir=$(mktemp -d)
mlp_proto=
seed=777
learn_rate=0.004
momentum=0.9
minibatch_size=256
randomizer_size=32768
error_function="fer"
train_tool="snnet-train-shuff"
test_tool="snnet-best"
dnn_depth=1
dnn_width=200
lattice_N=1000
negative_num=$((lattice_N*2))
acwt=0.2
train_opt=
cpus=$(nproc)
# End configuration.

echo "$0 $@"  # Print the command line for logging

. parse_options.sh || exit 1;

files="train.lab dev.lab test.lab train.ark dev.ark test.ark train.lat dev.lat test.lat"

if [ "$#" -ne 3 ]; then
   echo "Perform structure DNN training"
   echo "Usage: $0 <dir> <lattice_model> <model_out>"
   echo "eg. $0 data/nn_post model.in model.out"
   echo ""
   exit 1;
fi

dir=$1
lat_model=$2
model_out=$3

#check file existence.
for file in $files;
do
   [ -f $dir/$file ] || ( echo "File '$dir/$file' not found." && exit 1 );
done

train_ark="ark:$dir/train.ark"
train_lab="ark:$dir/train.lab"
train_lat="ark:$dir/train.lat"

cv_ark="ark:$dir/dev.ark"
cv_lab="ark:$dir/dev.lab"
cv_lat="ark:$dir/dev.lat"

lattice_N_times=$((lattice_N))


mkdir $tmpdir/nnet
mkdir $tmpdir/log

#initialize model
max_state=$(copy-int-vector "$train_lab" ark,t:-| cut -f 2- -d ' ' | tr " " "\n" | awk 'n < $0 {n=$0}END{print n}')
feat_dim=$(feat-to-dim "$train_ark" -)
SVM_dim=$(( (max_state + feat_dim) * max_state ))
mlp_init=$tmpdir/nnet.init
mlp_proto=$tmpdir/nnet.proto
$timit/utils/nnet/make_nnet_proto.py $SVM_dim 2 $dnn_depth $dnn_width > $mlp_proto || exit 1
nnet-initialize $mlp_proto $mlp_init || exit 1; 

# precompute lattice data
train_lattice_path=$dir/train.lab_${lattice_N_times}_${acwt}.gz

# lock file if others are using...
while [ ! -f $train_lattice_path ]; do
    lockfile=/tmp/$(basename $train_lattice_path)
    flock -n $lockfile \
       lattice-to-nbest-path.sh --cpus $cpus --acoustic-scale $acwt --n $lattice_N_times \
       $lat_model "$train_lat" "ark:| gzip -c > $train_lattice_path" \
       2>&1 | tee $log  || \
       flock -w -1 $lockfile echo "finally get file lock"
done

dev_lattice_path=$dir/dev.lab_${lattice_N_times}_${acwt}.gz
while [ ! -f $dev_lattice_path ]; do
    lockfile=/tmp/$(basename $dev_lattice_path)
    flock -n $lockfile \
       lattice-to-nbest-path.sh --cpus $cpus --acoustic-scale $acwt --n $lattice_N_times \
       $lat_model "$cv_lat" "ark:| gzip -c > $dev_lattice_path" \
       2>&1 | tee $log  || \
       flock -w -1 $lockfile echo "finally get file lock"
done


mlp_best=$mlp_init
#TODO use lattice best path as init path to do gibbs

loss=0
halving=0
# start training
for iter in $(seq -w $max_iters); do

   mlp_next=$tmpdir/nnet/nnet.${iter}

   # find negitive example
   log=$tmpdir/log/iter${iter}.ptr.log; hostname>$log

   train_lattice_path_rand=$dir/train.lab_${lattice_N}_${acwt}_${seed}.gz

   while [ ! -f $train_lattice_path_rand ]; do
       lockfile=/tmp/$(basename $train_lattice_path_rand)
       flock -n $lockfile \
          lattice-to-nbest-path.sh --cpus $cpus --acoustic-scale $acwt --random true --srand $seed \
          --n $lattice_N $lat_model "$train_lat" "ark:| gzip -c > $train_lattice_path_rand" \
          2>&1 | tee $log  || \
          flock -w -1 $lockfile echo "finally get file lock"
   done


   seed=$((seed + 1))


# train
   log=$tmpdir/log/iter${iter}.tr.log; hostname>$log

   $train_tool \
      --learn-rate=$learn_rate --momentum=$momentum --l1-penalty=$l1_penalty --l2-penalty=$l2_penalty \
      --minibatch-size=$minibatch_size --randomizer-size=$randomizer_size --randomize=true \
      --verbose=$verbose --binary=true --randomizer-seed=$seed \
      --negative-num=$negative_num --error-function=$error_function \
      ${train_opt:+ "$train_opt"} \
      "$train_ark" "$train_lab" "ark:combine-score-path ark:- \"ark:gunzip -c $train_lattice_path_rand |\" \"ark:gunzip -c $train_lattice_path |\" |" \
      $mlp_best $mlp_next \
      2>&1 | tee -a $log ; ( exit ${PIPESTATUS[0]} ) || exit 1;

      seed=$((seed + 1))
   loss_tr=$(cat $log | grep "AvgLoss:" | tail -n 1 | awk '{ print $4; }')
   echo -n "TRAIN AVG.LOSS $(printf "%.4f" $loss_tr), "

   log=$tmpdir/log/iter${iter}.cv.log; hostname>$log

   $test_tool "$cv_ark" "ark:gunzip -c $dev_lattice_path |" $mlp_next ark:$tmpdir/cv.ark\
      2>&1 | tee -a $log ; ( exit ${PIPESTATUS[0]} ) || exit 1;

   compute-wer "ark:trim-path $cv_lab ark:- |" "ark:split-score-path ark:$tmpdir/cv.ark ark:/dev/null ark:- | trim-path ark:- ark:- |" \
      2>&1 | tee -a $log ; ( exit ${PIPESTATUS[0]} ) || exit 1;

   loss_new=$(cat $log | grep 'WER' | tail -n 1 | awk '{ print $2; }')
   echo -n "CROSSVAL WER= $(printf "%.4f" $loss_new), "

   # accept or reject new parameters (based on objective function)
   loss_prev=$loss
   if [ 1 == $(bc <<< "$loss_new < $loss") -o $iter -le $keep_lr_iters ]; then
      loss=$loss_new
      mlp_best=$tmpdir/nnet/nnet.${iter}_learnrate${learn_rate}_tr$(printf "%.4f" $tr_loss)_cv$(printf "%.4f" $loss_new)
      [ $iter -le $keep_lr_iters ] && mlp_best=${mlp_best}_keep-lr-iters-$keep_lr_iters
      mv $mlp_next $mlp_best
      echo "nnet accepted ($(basename $mlp_best))"
   else
     mlp_reject=$tmpdir/nnet/nnet.${iter}_learnrate${learn_rate}_tr$(printf "%.4f" $tr_loss)_cv$(printf "%.4f" $loss_new)_rejected
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
#rm -f $tmpdir/train_tmp.ark
#rm -f $tmpdir/train.lab
rm -rf $tmpdir


exit 0;



