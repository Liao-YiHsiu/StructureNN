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
minibatch_size=8
randomizer_size=4194304
error_function="fer"
train_tool="snnet-train-fullshuff"
cross_tool="snnet-train-cross"
#train_tool="snnet-train-shuff"
#cross_tool="snnet-train-shuff --cross-validate=true"
dnn_depth=1
dnn_width=200
lattice_N=1000
test_lattice_N=1000
acwt=0.2
train_opt=
cpus=$(nproc)
feature_transform=
nnet_ratio=
# End configuration.

echo "$0 $@"  # Print the command line for logging

. parse_options.sh || exit 1;

negative_num=$((lattice_N))

files="train.lab dev.lab test.lab train.ark dev.ark test.ark train.lat dev.lat test.lat nnet1"

if [ "$#" -ne 5 ]; then
   echo "Perform structure DNN training"
   echo "Usage: $0 <dir> <lattice_model> <nnet1-out> <nnet2-out> <stateMax>"
   echo "eg. $0 data/nn_post lat_model.in nnet1.out nnet2.out 48"
   echo ""
   exit 1;
fi

dir=$1
lat_model=$2
nnet1_out=$3
nnet2_out=$4
stateMax=$5

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

mkdir $tmpdir/nnet
mkdir $tmpdir/log

#initialize nnet2 model
feat_dim=$(nnet-info $dir/nnet1 2>/dev/null | grep output-dim | head -n 1 | cut -d ' ' -f2)
SVM_dim=$(( (stateMax + 1) * (feat_dim + 1) + stateMax*stateMax ))
mlp2_init=$tmpdir/nnet2.init
mlp_proto=$tmpdir/nnet.proto
$timit/utils/nnet/make_nnet_proto.py --no-softmax $SVM_dim 1 $dnn_depth $dnn_width | \
   sed -e "s@</NnetProto>@<Sigmoid> <InputDim> 1 <OutputDim> 1 \n</NnetProto>@g" > $mlp_proto || exit 1
nnet-initialize $mlp_proto $mlp2_init || exit 1; 

# precompute lattice data
train_lattice_path=$dir/train.lab_${lattice_N}_${acwt}.gz

# lock file if others are using...
while [ ! -f $train_lattice_path ]; do
    lockfile=/tmp/$(basename $train_lattice_path)
    flock -n $lockfile \
       lattice-to-nbest-path.sh --cpus $cpus --acoustic-scale $acwt --n $lattice_N \
       $lat_model "$train_lat" "ark:| gzip -c > $train_lattice_path" \
       2>&1 | tee $log  || \
       flock -w -1 $lockfile echo "finally get file lock"
done

dev_lattice_path=$dir/dev.lab_${test_lattice_N}_${acwt}.gz
while [ ! -f $dev_lattice_path ]; do
    lockfile=/tmp/$(basename $dev_lattice_path)
    flock -n $lockfile \
       lattice-to-nbest-path.sh --cpus $cpus --acoustic-scale $acwt --n $test_lattice_N \
       $lat_model "$cv_lat" "ark:| gzip -c > $dev_lattice_path" \
       2>&1 | tee $log  || \
       flock -w -1 $lockfile echo "finally get file lock"
done

mlp1_best=$dir/nnet1
mlp2_best=$mlp2_init

   snnet-score ${feature_transform:+ --feature-transform="$feature_transform"} "$cv_ark" \
      "ark:gunzip -c $dev_lattice_path |" $mlp1_best $mlp2_best $stateMax \
      "ark:| gzip -c > $tmpdir/dev_${iter}.tag.gz"\
      2>&1 | tee -a $log ; ( exit ${PIPESTATUS[0]} ) || exit 1;

   best-score-path "ark:gunzip -c $tmpdir/dev_${iter}.tag.gz |" ark:$tmpdir/dev_${iter}.tag.1best
      2>&1 | tee -a $log ; ( exit ${PIPESTATUS[0]} ) || exit 1;

   calc.sh "$cv_lab" ark:$tmpdir/dev_${iter}.tag.1best \
      2>&1 | tee -a $log ; ( exit ${PIPESTATUS[0]} ) || exit 1;


#TODO use lattice best path as init path to do gibbs

loss=0
halving=0
# start training
for iter in $(seq -w $max_iters); do

   mlp1_next=$tmpdir/nnet/nnet1.${iter}
   mlp2_next=$tmpdir/nnet/nnet2.${iter}

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
   
# retry on fail
   retry=10
   while [ $retry -gt 0 ]; do
      flock -n /tmp/gpu sleep 5
      $train_tool \
         --learn-rate=$learn_rate --momentum=$momentum --l1-penalty=$l1_penalty --l2-penalty=$l2_penalty \
         --minibatch-size=$minibatch_size --randomizer-size=$randomizer_size --randomize=true \
         --verbose=$verbose --binary=true --randomizer-seed=$seed \
         --negative-num=$negative_num --error-function=$error_function \
         ${feature_transform:+ --feature-transform="$feature_transform"} \
         ${nnet_ratio:+ --nnet-ratio="$nnet_ratio"} \
         ${train_opt:+ "$train_opt"} \
         "$train_ark" "$train_lab" "ark:combine-score-path ark:- \"ark:gunzip -c $train_lattice_path_rand |\" \"ark:gunzip -c $train_lattice_path |\" |" \
         $mlp1_best $mlp2_best $stateMax $mlp1_next $mlp2_next \
         2>&1 | tee -a $log ; ( exit ${PIPESTATUS[0]} ) && break;
         #2>&1 | grep  --line-buffered -v "releasing cached memory and retrying" | tee -a $log ; ( exit ${PIPESTATUS[0]} ) && break;
      retry=$((retry - 1))
      sleep 3
   done
   [ $retry -eq 0 ] && echo "retry over 10 times" && exit -1;

   seed=$((seed + 1))
   loss_tr=$(cat $log | grep "AvgLoss:" | tail -n 1 | awk '{ print $4; }')
   echo -n "TRAIN AVG.LOSS $(printf "%.4f" $loss_tr), "

   log=$tmpdir/log/iter${iter}.cv.log; hostname>$log
   
# CROSS VALIDATION
# retry on fail
   retry=10
   while [ $retry -gt 0 ]; do
      flock -n /tmp/gpu sleep 5
      $cross_tool \
         --verbose=$verbose --binary=true --error-function=$error_function \
         ${feature_transform:+ --feature-transform="$feature_transform"} \
         "$cv_ark" "$cv_lab" "ark:gunzip -c $dev_lattice_path |" \
         $mlp1_next $mlp2_next $stateMax \
         2>&1 | tee -a $log ; ( exit ${PIPESTATUS[0]} ) && break;
      retry=$((retry - 1))
      sleep 3
   done
   [ $retry -eq 0 ] && echo "retry over 10 times" && exit -1;

   loss_new=$(cat $log | grep "AvgLoss:" | tail -n 1 | awk '{ print $4; }')
   echo -n "CROSSVAL AVG.LOSS $(printf "%.4f" $loss_new), "

# evaluate dev set wer.
# ------------------------------------------------------------------------------------------------
   snnet-score ${feature_transform:+ --feature-transform="$feature_transform"} "$cv_ark" \
      "ark:gunzip -c $dev_lattice_path |" $mlp1_next $mlp2_next $stateMax \
      "ark:| gzip -c > $tmpdir/dev_${iter}.tag.gz"\
      2>&1 | tee -a $log ; ( exit ${PIPESTATUS[0]} ) || exit 1;

   best-score-path "ark:gunzip -c $tmpdir/dev_${iter}.tag.gz |" ark:$tmpdir/dev_${iter}.tag.1best
      2>&1 | tee -a $log ; ( exit ${PIPESTATUS[0]} ) || exit 1;

   calc.sh "$cv_lab" ark:$tmpdir/dev_${iter}.tag.1best \
      2>&1 | tee -a $log ; ( exit ${PIPESTATUS[0]} ) || exit 1;
# ------------------------------------------------------------------------------------------------

   # accept or reject new parameters (based on objective function)
   loss_prev=$loss
   if [ 1 == $(bc <<< "$loss_new < $loss") -o $iter -le $keep_lr_iters ]; then
      loss=$loss_new
      mlp1_best=$tmpdir/nnet/nnet1.${iter}_learnrate${learn_rate}_tr$(printf "%.4f" $tr_loss)_cv$(printf "%.4f" $loss_new)
      mlp2_best=$tmpdir/nnet/nnet2.${iter}_learnrate${learn_rate}_tr$(printf "%.4f" $tr_loss)_cv$(printf "%.4f" $loss_new)
      [ $iter -le $keep_lr_iters ] && mlp1_best=${mlp1_best}_keep-lr-iters-$keep_lr_iters 
      [ $iter -le $keep_lr_iters ] && mlp2_best=${mlp2_best}_keep-lr-iters-$keep_lr_iters

      mv $mlp1_next $mlp1_best
      mv $mlp2_next $mlp2_best
      echo "nnet accepted ($(basename $mlp1_best) $(basename $mlp2_best))"
   else
     mlp1_reject=$tmpdir/nnet/nnet1.${iter}_learnrate${learn_rate}_tr$(printf "%.4f" $tr_loss)_cv$(printf "%.4f" $loss_new)_rejected
     mlp2_reject=$tmpdir/nnet/nnet2.${iter}_learnrate${learn_rate}_tr$(printf "%.4f" $tr_loss)_cv$(printf "%.4f" $loss_new)_rejected
     mv $mlp1_next $mlp1_reject
     mv $mlp2_next $mlp2_reject
     echo "nnet rejected ($(basename $mlp1_reject) $(basename $mlp2_reject))"
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

cp $mlp1_best $nnet1_out
cp $mlp2_best $nnet2_out
#rm -f $tmpdir/train_tmp.ark
#rm -f $tmpdir/train.lab
rm -rf $tmpdir


exit 0;



