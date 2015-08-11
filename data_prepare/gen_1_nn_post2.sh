#!/bin/bash -ex

# no softmax layer

DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
source $DIR/path.sh
cd $DIR/../data/nn_post2

dim=49
mlp_final="final.nnet"

if [ ! -f $mlp_final ]; then
   # init NN
   mlp_init="net.init"
   mlp_proto="net.proto"
   mlp_old="net.old"
   mlp_new="net.new"
   mlp_train="net.train"

   # remove softmax layer from the original nnet
   nnet-pop --num=2 $timit/exp/dnn4_pretrain-dbn_dnn_smbr/final.nnet $mlp_old
   feat_dim=$(nnet-info $mlp_old  2>/dev/null | grep output-dim | tail -n 1 | sed -E "s/^.*output-dim//g" | sed -E "s/[^0-9]//g")

   $timit/utils/nnet/make_nnet_proto.py $feat_dim $dim 0 1 > $mlp_proto || exit 1
   nnet-initialize $mlp_proto $mlp_init || exit 1; 

   nnet-concat $mlp_old $mlp_init $mlp_new

   # train NN label
   train.sh --feature-transform "$timit/exp/dnn4_pretrain-dbn_dnn_smbr/final.feature_transform" $mlp_new "$source_tr" "ark:ali-to-post ark:train32.lab ark:- |" "$source_dv" "ark:ali-to-post ark:dev32.lab ark:- |" $mlp_train || exit 1

   #remove softmax layer
   nnet-pop --num=1 $mlp_train $mlp_final
fi

$nn_forward $mlp_final $source_tr ark:- | replace-feats ark:- ark:train.ark 0 1.0 || exit 1;
$nn_forward $mlp_final $source_dv ark:- | replace-feats ark:- ark:dev.ark   0 1.0 || exit 1;
$nn_forward $mlp_final $source_ts ark:- | replace-feats ark:- ark:test.ark  0 1.0 || exit 1;
