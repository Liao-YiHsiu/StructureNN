#!/bin/bash -ex
# kill child process upon exit

DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
source $DIR/../path

# Begin configuration.
config=

# training options
l1_penalty=0
l2_penalty=0
max_iters=10000
min_iters=
keep_lr_iters=1
start_halving_impr=0.005
end_halving_impr=0.00005
halving_factor=0.5

verbose=1

frame_weights=
seed=777
learn_rate=0.004
momentum=0
minibatch_size=64
randomizer_size=4194304
train_tool="snnet-train-pairshuff "
dnn_depth=1
dnn_width=200
lattice_N=1000
test_lattice_N=1000
acwt=0.2
train_opt=
cpus=$(nproc)
feature_transform=
rbm_pretrain="true"
lattice_source="both" # both, best, rand
tmpdir=$(mktemp -d)
debug=
# End configuration.

echo "$0 $@"  # Print the command line for logging

. parse_options.sh || exit 1;

if [ "$debug" == "true" ]; then
   trap 'kill $(jobs -p) || true ' EXIT
else
   trap 'rm -rf $tmpdir; kill $(jobs -p) || true ' EXIT
fi

cross_tool="$train_tool --cross-validate=true"

#negative_num=$((lattice_N))
negative_num=0

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

nnetdir=$(cd `dirname $nnet1_out`; pwd)

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

mkdir -p $tmpdir/nnet
mkdir -p $tmpdir/log

# precompute lattice data
train_lattice_path=$dir/../lab/train.lab_${lattice_N}_${acwt}.gz
lattice-to-nbest-path.sh --cpus $(( (cpus + 1)/2 )) --acoustic-scale $acwt --n $lattice_N \
   $lat_model "$train_lat" "$train_lattice_path" \
   2>&1 | tee -a $log; ( exit ${PIPESTATUS[0]} ) || exit 1

dev_lattice_path=$dir/../lab/dev.lab_${test_lattice_N}_${acwt}.gz
lattice-to-nbest-path.sh --cpus $(( (cpus + 1)/2 )) --acoustic-scale $acwt --n $test_lattice_N \
   $lat_model "$cv_lat" "$dev_lattice_path" \
   2>&1 | tee -a $log; ( exit ${PIPESTATUS[0]} ) || exit 1

test_lattice_path=$dir/../lab/test.lab_${test_lattice_N}_${acwt}.gz
lattice-to-nbest-path.sh --cpus $(( (cpus+1)/2 )) --acoustic-scale $acwt --n $test_lattice_N \
   $lat_model ark:$dir/test.lat "$test_lattice_path" \
   2>&1 | tee -a $log ; ( exit ${PIPESTATUS[0]} ) || exit 1;

# use pretrained front end neural network
mlp1_best=$dir/nnet1
mlp2=$tmpdir/nnet2
mlp2_best=$mlp2

if [ $rbm_pretrain == "true" ]; then
   # RBM pretraining
   [ ! -d $dir/../psi ] && mkdir $dir/../psi
   psi_feature=$(readlink -e $dir)/../psi/train_${lattice_N}_${acwt}

   if [ ! -f ${psi_feature}.scp ]; then
      snnet-gen-psi \
         --negative-num=$negative_num \
         --rand-seed=$seed \
         ${feature_transform:+ --feature-transform="$feature_transform"} \
         "$train_ark" "$train_lab" "ark:gunzip -c $train_lattice_path |" \
         $mlp1_best $stateMax ark,scp:${psi_feature}.ark,${psi_feature}.scp \
         2>&1 | tee -a $log; ( exit ${PIPESTATUS[0]} ) || exit 1
   fi

   mkdir -p $tmpdir/psi
   cp ${psi_feature}.scp $tmpdir/psi/feats.scp

   dbn=$dir/${dnn_depth}_${dnn_width}_${lattice_N}_${acwt}.dbn
   dbn_transform=$dir/${dnn_depth}_${dnn_width}_${lattice_N}_${acwt}.trans

   if [ ! -f $dbn ]; then
      (cd $timit && steps/nnet/pretrain_dbn.sh \
         --nn-depth $((dnn_depth - 1)) --hid_dim $dnn_width \
         --splice 0 --rbm-iter 20 \
         $tmpdir/psi $tmpdir/dbn \
         2>&1 | tee -a $log);
      cp $tmpdir/dbn/$((dnn_depth - 1)).dbn $dbn || exit 1
      cp $tmpdir/dbn/final.feature_transform $dbn_transform || exit 1
   fi

   #initialize nnet2 model
   mlp2_last=$tmpdir/nnet2.last
   mlp2_proto=$tmpdir/nnet2.proto
   $timit/utils/nnet/make_nnet_proto.py --no-softmax $dnn_width 1 0 $dnn_width > $mlp2_proto || exit 1
   nnet-initialize $mlp2_proto $mlp2_last || exit 1; 

   # remove splice in dbn_transform.
   nnet-copy --binary=false $dbn_transform - | \
      sed  -e 'N;s@<Splice\>.*@@g' -e 's@\[ 0 \]@@g' | \
      sed '/^$/d' > $tmpdir/nosplice

   nnet-concat $tmpdir/nosplice $dbn $mlp2_last $mlp2 \
      2>&1 | tee -a $log ; ( exit ${PIPESTATUS[0]} ) || exit 1

else 

   #initialize nnet2 model
   feat_dim=$(nnet-info $dir/nnet1 2>/dev/null | grep output-dim | head -n 1 | cut -d ' ' -f2)
   SVM_dim=$(( (stateMax + 1) * (feat_dim + 1) + stateMax*stateMax ))
   mlp2_init=$tmpdir/nnet2.init
   mlp2_proto=$tmpdir/nnet2.proto
   $timit/utils/nnet/make_nnet_proto.py --no-softmax $SVM_dim 1 $dnn_depth $dnn_width | \
      sed -e "s@</NnetProto>@</NnetProto>@g" > $mlp2_proto || exit 1
   #   add Sigmoid layer in the last layer
   #   sed -e "s@</NnetProto>@<Sigmoid> <InputDim> 1 <OutputDim> 1 \n</NnetProto>@g" > $mlp2_proto || exit 1
   nnet-initialize $mlp2_proto $mlp2_init || exit 1; 

   mlp2_cmvn=$tmpdir/nnet2.cvmn
   mlp2=$tmpdir/nnet2

   #compute cmvn layer for nnet2
   snnet-psi-cmvn --verbose=$verbose --binary=true --rand-seed=$seed --negative-num=$negative_num \
      ${feature_transform:+ --feature-transform="$feature_transform"} \
      "$train_ark" "$train_lab" "ark:gunzip -c $train_lattice_path |" \
      $dir/nnet1 $mlp2_init $stateMax $mlp2_cmvn \
      2>&1 | tee -a $log ; ( exit ${PIPESTATUS[0]} ) || exit 1

   nnet-concat $mlp2_cmvn $mlp2_init $mlp2 \
      2>&1 | tee -a $log ; ( exit ${PIPESTATUS[0]} ) || exit 1

fi



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
   seed=$((seed + 1))

   # find negitive example
   log=$tmpdir/log/iter${iter}.ptr.log; hostname>$log

   if [ "$lattice_source" == "both" -o "$lattice_source" == "rand" ]; then
      train_lattice_path_rand=$dir/../lab/train.lab_${lattice_N}_${acwt}_${seed}.gz
      lattice-to-nbest-path.sh --cpus $(( (cpus + 1)/2 )) --acoustic-scale $acwt \
         --random true --srand $seed \
         --n $lattice_N $lat_model "$train_lat" "$train_lattice_path_rand" \
         2>&1 | tee -a $log ; ( exit ${PIPESTATUS[0]} ) || exit 1;

      # precompute next iteration in backgroud
      seed_next=$((seed + 1))
      train_lattice_path_rand_next=$dir/../lab/train.lab_${lattice_N}_${acwt}_${seed_next}.gz
      lattice-to-nbest-path.sh --cpus $(( (cpus + 1)/2 )) --acoustic-scale $acwt \
         --random true --srand $seed_next \
         --n $lattice_N $lat_model "$train_lat" "$train_lattice_path_rand_next" \
         >/dev/null 2>/dev/null &
   fi

   # train
   log=$tmpdir/log/iter${iter}.tr.log; hostname>$log

   if [ "$lattice_source" == "both" ]; then
      combined_score_path="ark:combine-score-path ark:- \"ark:gunzip -c $train_lattice_path_rand |\" \"ark:gunzip -c $train_lattice_path |\" |"

   elif [ "$lattice_source" == "best" ] ; then
      combined_score_path="ark:gunzip -c $train_lattice_path |"

   elif [ "$lattice_source" == "rand" ] ; then
      combined_score_path="ark:gunzip -c $train_lattice_path_rand |"

   else
      echo "lattice-source error, lattice-source=$lattice_source"
      exit -1;
   fi

   $train_tool \
      --learn-rate=$learn_rate --momentum=$momentum --l1-penalty=$l1_penalty --l2-penalty=$l2_penalty \
      --minibatch-size=$minibatch_size --randomizer-size=$randomizer_size --randomize=true \
      --verbose=$verbose --binary=true --randomizer-seed=$seed \
      --negative-num=$negative_num \
      ${feature_transform:+ --feature-transform="$feature_transform"} \
      ${train_opt:+ "$train_opt"} \
      "$train_ark" "$train_lab" "$combined_score_path" \
      $mlp1_best $mlp2_best $stateMax $mlp1_next $mlp2_next \
      2>&1 | tee -a $log ; ( exit ${PIPESTATUS[0]} ) || exit 1;

   loss_tr=$(cat $log | grep "AvgLoss:" | tail -n 1 | awk '{ print $4; }')
   acc_tr=$(cat $log | grep "FRAME_ACCURACY" | tail -n 1 | awk '{ print $3; }')
   echo -n "TRAIN AVG.LOSS $(printf "%.4f" $loss_tr), "

   log=$tmpdir/log/iter${iter}.cv.log; hostname>$log

   # CROSS VALIDATION
   $cross_tool \
      --verbose=$verbose --binary=true\
      --minibatch-size=$minibatch_size \
      ${feature_transform:+ --feature-transform="$feature_transform"} \
      "$cv_ark" "$cv_lab" "ark:gunzip -c $dev_lattice_path |" \
      $mlp1_next $mlp2_next $stateMax \
      2>&1 | tee -a $log ; ( exit ${PIPESTATUS[0]} ) || exit 1;

   #loss_new=$(cat $log | grep "FRAME_ACCURACY" | tail -n 1 | awk '{print 100 - $3}')
   loss_new=$(cat $log | grep "AvgLoss:" | tail -n 1 | awk '{ print $4; }')
   acc_new=$(cat $log | grep "FRAME_ACCURACY" | tail -n 1 | awk '{ print $3; }')
   echo -n "CROSSVAL AVG.LOSS $(printf "%.4f" $loss_new), "

# evaluate dev set wer.
# ------------------------------------------------------------------------------------------------
#   snnet-score ${feature_transform:+ --feature-transform="$feature_transform"} "$cv_ark" \
#      "ark:gunzip -c $dev_lattice_path |" $mlp1_next $mlp2_next $stateMax \
#      "ark:| gzip -c > $tmpdir/dev_${iter}.tag.gz"\
#      2>&1 | tee -a $log ; ( exit ${PIPESTATUS[0]} ) || exit 1;
#
#   best-score-path "ark:gunzip -c $tmpdir/dev_${iter}.tag.gz |" ark:$tmpdir/dev_${iter}.tag.1best
#      2>&1 | tee -a $log ; ( exit ${PIPESTATUS[0]} ) || exit 1;
#
#   calc.sh "$cv_lab" ark:$tmpdir/dev_${iter}.tag.1best \
#      2>&1 | tee -a $log ; ( exit ${PIPESTATUS[0]} ) || exit 1;
#   WER=$(grep WER $log | tail -n 1 | awk '{ print $2; }')
# ------------------------------------------------------------------------------------------------


   snnet-score ${feature_transform:+ --feature-transform="$feature_transform"} ark:$dir/test.ark \
      "ark:gunzip -c $test_lattice_path |" $mlp1_next $mlp2_next $stateMax \
      "ark:| gzip -c > $tmpdir/test_${iter}.tag.gz"\
      2>&1 | tee -a $log ; ( exit ${PIPESTATUS[0]} ) || exit 1;

   best-score-path "ark:gunzip -c $tmpdir/test_${iter}.tag.gz |" ark:$tmpdir/test_${iter}.tag.1best
      2>&1 | tee -a $log ; ( exit ${PIPESTATUS[0]} ) || exit 1;

   calc.sh ark:$dir/test.lab ark:$tmpdir/test_${iter}.tag.1best \
      2>&1 | tee -a $log ; ( exit ${PIPESTATUS[0]} ) || exit 1;
   WER=$(grep WER $log | tail -n 1 | awk '{ print $2; }')

   # accept or reject new parameters (based on objective function)
   loss_prev=$loss
   
   echo -n "$iter $(printf '%.10f' $learn_rate) $loss_tr $loss_new $acc_tr $acc_new $WER " >> $nnetdir/log

   if [ 1 == $(bc <<< "${loss_new/[eE]+*/*10^} < ${loss/[eE]+*/*10^}") -o $iter -le $keep_lr_iters ]; then
      loss=$loss_new
      mlp1_best=$tmpdir/nnet/nnet1.${iter}_learnrate${learn_rate}_tr$(printf "%.4f" $loss_tr)_acc${acc_tr}_cv$(printf "%.4f" $loss_new)_acc${acc_new}_wer${WER}
      mlp2_best=$tmpdir/nnet/nnet2.${iter}_learnrate${learn_rate}_tr$(printf "%.4f" $loss_tr)_acc${acc_tr}_cv$(printf "%.4f" $loss_new)_acc${acc_new}_wer${WER}
      echo "accept" >> $nnetdir/log
      [ $iter -le $keep_lr_iters ] && mlp1_best=${mlp1_best}_keep-lr-iters-$keep_lr_iters 
      [ $iter -le $keep_lr_iters ] && mlp2_best=${mlp2_best}_keep-lr-iters-$keep_lr_iters

      mv $mlp1_next $mlp1_best
      mv $mlp2_next $mlp2_best
      echo "nnet accepted ($(basename $mlp1_best) $(basename $mlp2_best))"
   else
     mlp1_reject=$tmpdir/nnet/nnet1.${iter}_learnrate${learn_rate}_tr$(printf "%.4f" $loss_tr)_acc${acc_tr}_cv$(printf "%.4f" $loss_new)_acc${acc_new}_wer${WER}_rejected
     mlp2_reject=$tmpdir/nnet/nnet2.${iter}_learnrate${learn_rate}_tr$(printf "%.4f" $loss_tr)_acc${acc_tr}_cv$(printf "%.4f" $loss_new)_acc${acc_new}_wer${WER}_rejected
     echo "rejected" >> $nnetdir/log
     mv $mlp1_next $mlp1_reject
     mv $mlp2_next $mlp2_reject
     echo "nnet rejected ($(basename $mlp1_reject) $(basename $mlp2_reject))"
   fi

   # no learn-rate halving yet, if keep_lr_iters set accordingly
   [ $iter -le $keep_lr_iters ] && continue 

   # stopping criterion
   rel_impr=$(bc <<< "scale=10; (${loss_prev/[eE]+*/*10^}-${loss/[eE]+*/*10^})/${loss_prev/[eE]+*/*10^}")
   if [ 1 == $halving -a 1 == $(bc <<< "${rel_impr/[eE]+*/*10^} < ${end_halving_impr/[eE]+*/*10^}") ]; then
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
   if [ 1 == $(bc <<< "${rel_impr/[eE]+*/*10^} < ${start_halving_impr/[eE]+*/*10^}") ]; then
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

rm -rf $tmpdir

exit 0;


