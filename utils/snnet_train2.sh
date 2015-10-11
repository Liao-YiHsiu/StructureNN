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
randomizer_size=4194304
train_tool="srnnet-train-pairshuff "
dnn_depth=1
dnn_width=256
rnn_width=64
test_lattice_N=1000
acwt=0.2
train_opt=
cpus=$(nproc)
feature_transform=
tmpdir=$(mktemp -d)
debug=

dnn1_depth=
dnn1_width=

# End configuration.

echo "$0 $@"  # Print the command line for logging

. parse_options.sh || exit 1;

if [ "$debug" == "true" ]; then
   trap 'kill $(jobs -p) || true ' EXIT
else
   trap 'rm -rf $tmpdir; kill $(jobs -p) || true ' EXIT
fi

cross_tool="$train_tool --cross-validate=true "
train_tool="$train_tool "

files="train.lab dev.lab test.lab train.ark dev.ark test.ark train.lat dev.lat test.lat nnet1"

if [ "$#" -ne 4 ]; then
   echo "Perform structure DNN training"
   echo "Usage: $0 <dir> <lattice_model> <nnet> <stateMax>"
   echo "eg. $0 data/nn_post lat_model.in lat_model.in nnet.out 48"
   echo ""
   exit 1;
fi

dir=$1
lat_model=$2
nnet_out=$3
stateMax=$4

nnetdir=$(cd `dirname $nnet_out`; pwd)

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
dev_lattice_path=$dir/../lab/dev.lab_${test_lattice_N}_${acwt}.gz
lattice-to-nbest-path.sh --cpus $(( (cpus + 1)/2 )) --acoustic-scale $acwt --n $test_lattice_N \
   $lat_model "$cv_lat" "$dev_lattice_path" \
   2>&1 | tee -a $log; ( exit ${PIPESTATUS[0]} ) || exit 1

# use pretrained front end neural network
mlp2=$tmpdir/nnet2
mlp1_best=$dir/nnet1

if [ ! -z $dnn1_depth ] && [ ! -z $dnn1_width ]; then
   feat_in_dim=$(nnet-info $mlp1_best 2>/dev/null | grep input-dim | head -n 1 | cut -d ' ' -f2)
   mlp1_best=$tmpdir/nnet1

   mlp1_proto=$tmpdir/nnet1.proto
   mlp1_init=$tmpdir/nnet1.init
   $timit/utils/nnet/make_nnet_proto.py $feat_in_dim $dnn1_width $((dnn1_depth + 1)) $dnn1_width > $mlp1_proto  \
      || exit 1
   nnet-initialize $mlp1_proto $mlp1_init || exit 1; 
   nnet-pop --num=2 $mlp1_init $mlp1_best || exit 1;
fi

mlp2_best=$mlp2
mlp_best=$tmpdir/nnet.init

   #initialize nnet2 model
   feat_dim=$(nnet-info $mlp1_best 2>/dev/null | grep output-dim | head -n 1 | cut -d ' ' -f2)

   mlp2_proto=$tmpdir/nnet2.proto
   $timit/utils/nnet/make_nnet_proto.py --no-softmax $rnn_width 1 $dnn_depth $dnn_width > $mlp2_proto  \
      || exit 1
   nnet-initialize $mlp2_proto $mlp2 || exit 1; 

   #generate configure file
   srnn_config=$tmpdir/srnn.config
   stddev=$(echo "scale=10; 3.5 * sqrt(2/($feat_dim + $rnn_width))" | bc)
   Affine="<AffineTransform> <InputDim> $feat_dim <OutputDim> $rnn_width <BiasMean> 0.0 <BiasRange> 4.0 <ParamStddev> 0.02 <MaxNorm> 0.0"

   echo "<SRNnetProto>" > $srnn_config
   echo "<MUX>"        >> $srnn_config

   for((i=0; i < $stateMax; ++i))
   do
      echo $Affine >> $srnn_config
   done

   cat >>$srnn_config <<EOF
   </MUX>
   <Init> <AddShift> <InputDim> $rnn_width <OutputDim> $rnn_width <InitParam> 0.0
   <Forw> <AffineTransform> <InputDim> $rnn_width <OutputDim> $rnn_width <BiasMean> 0.0 <BiasRange> 4.0 <ParamStddev> 0.02 <MaxNorm> 0.0
   <Acti> <Sigmoid> <InputDim> $rnn_width <OutputDim> $rnn_width
   </SRNnetProto>
EOF
   
   srnnet-init $mlp1_best $mlp2_best $srnn_config $mlp_best


   srnnet-score ${feature_transform:+ --feature-transform="$feature_transform"} "$cv_ark" \
      "ark:gunzip -c $dev_lattice_path |" $mlp_best \
      "ark:| gzip -c > $tmpdir/dev_${iter}.tag.gz"\
      2>&1 | tee -a $log ; ( exit ${PIPESTATUS[0]} ) || exit 1;

   best-score-path "ark:gunzip -c $tmpdir/dev_${iter}.tag.gz |" ark:$tmpdir/dev_${iter}.tag.1best
      2>&1 | tee -a $log ; ( exit ${PIPESTATUS[0]} ) || exit 1;

   calc.sh "$cv_lab" ark:$tmpdir/dev_${iter}.tag.1best \
      2>&1 | tee -a $log ; ( exit ${PIPESTATUS[0]} ) || exit 1;

rm -f $nnetdir/log

loss=0
halving=0
# start training
for iter in $(seq -w $max_iters); do

   mlp_next=$tmpdir/nnet/nnet.${iter}
   seed=$((seed + 1))

   # train
   log=$tmpdir/log/iter${iter}.tr.log; hostname>$log

   $train_tool \
      --learn-rate=$learn_rate --momentum=$momentum --l1-penalty=$l1_penalty --l2-penalty=$l2_penalty \
      --randomizer-size=$randomizer_size --randomize=true \
      --verbose=$verbose --binary=true --randomizer-seed=$seed \
      ${feature_transform:+ --feature-transform="$feature_transform"} \
      ${train_opt:+ "$train_opt"} \
      "$train_ark" "$train_lab" \
      $mlp_best $mlp_next \
      2>&1 | tee -a $log ; ( exit ${PIPESTATUS[0]} ) || exit 1;

   loss_tr=$(cat $log | grep "AvgLoss:" | tail -n 1 | awk '{ print $4; }')
   acc_tr=$(cat $log | grep "FRAME_ACCURACY" | tail -n 1 | awk '{ print $3; }')
   echo -n "TRAIN AVG.LOSS $(printf "%.4f" $loss_tr), "

   log=$tmpdir/log/iter${iter}.cv.log; hostname>$log

   # CROSS VALIDATION
   $cross_tool \
      --verbose=$verbose --binary=true \
      --randomizer-size=$randomizer_size --randomize=true \
      ${feature_transform:+ --feature-transform="$feature_transform"} \
      "$cv_ark" "$cv_lab" $mlp_next \
      2>&1 | tee -a $log ; ( exit ${PIPESTATUS[0]} ) || exit 1;

   #loss_new=$(cat $log | grep "FRAME_ACCURACY" | tail -n 1 | awk '{print 100 - $3}')
   loss_new=$(cat $log | grep "AvgLoss:" | tail -n 1 | awk '{ print $4; }')
   acc_new=$(cat $log | grep "FRAME_ACCURACY" | tail -n 1 | awk '{ print $3; }')
   echo -n "CROSSVAL AVG.LOSS $(printf "%.4f" $loss_new), "

# evaluate dev set wer.
# ------------------------------------------------------------------------------------------------
   srnnet-score ${feature_transform:+ --feature-transform="$feature_transform"} "$cv_ark" \
      "ark:gunzip -c $dev_lattice_path |" $mlp_next\
      "ark:| gzip -c > $tmpdir/dev_${iter}.tag.gz"\
      2>&1 | tee -a $log ; ( exit ${PIPESTATUS[0]} ) || exit 1;
   
   best-score-path "ark:gunzip -c $tmpdir/dev_${iter}.tag.gz |" ark:$tmpdir/dev_${iter}.tag.1best
      2>&1 | tee -a $log ; ( exit ${PIPESTATUS[0]} ) || exit 1;

   calc.sh "$cv_lab" ark:$tmpdir/dev_${iter}.tag.1best \
      2>&1 | tee -a $log ; ( exit ${PIPESTATUS[0]} ) || exit 1;
   WER=$(grep WER $log | tail -n 1 | awk '{ print $2; }')
# ------------------------------------------------------------------------------------------------

   # accept or reject new parameters (based on objective function)
   loss_prev=$loss
   
   echo -n "$iter $(printf '%.10f' $learn_rate) $loss_tr $loss_new $acc_tr $acc_new $WER " >> $nnetdir/log

   if [ 1 == $(bc <<< "${loss_new/[eE]+*/*10^} < ${loss/[eE]+*/*10^}") -o $iter -le $keep_lr_iters ]; then
      loss=$loss_new
      mlp_best=$tmpdir/nnet/nnet.${iter}_learnrate${learn_rate}_tr$(printf "%.4f" $loss_tr)_acc${acc_tr}_cv$(printf "%.4f" $loss_new)_acc${acc_new}_wer${WER}
      echo "accept" >> $nnetdir/log
      [ $iter -le $keep_lr_iters ] && mlp_best=${mlp_best}_keep-lr-iters-$keep_lr_iters 

      mv $mlp_next $mlp_best
      echo "nnet accepted ($(basename $mlp_best))"
   else
     mlp_reject=$tmpdir/nnet/nnet.${iter}_learnrate${learn_rate}_tr$(printf "%.4f" $loss_tr)_acc${acc_tr}_cv$(printf "%.4f" $loss_new)_acc${acc_new}_wer${WER}_rejected
     echo "rejected" >> $nnetdir/log
     mv $mlp_next $mlp_reject
     echo "nnet rejected ($(basename $mlp_reject))"
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

cp $mlp_best $nnet_out

rm -rf $tmpdir

exit 0;


