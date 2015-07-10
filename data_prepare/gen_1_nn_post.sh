#!/bin/bash

DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
source $DIR/path.sh
cd $DIR/../data/nn_post

dim=49

if [ ! -f model.nnet ]; then
   # init NN
   mlp_init="net.init"
   mlp_proto="net.proto"

   feat_dim=$(feat-to-dim --print-args=false "$feats_ts" -)

   $timit/utils/nnet/make_nnet_proto.py $feat_dim $dim 0 1 > $mlp_proto || exit 1
   nnet-initialize $mlp_proto $mlp_init || exit 1; 

   # train NN label
   train.sh $mlp_init "$feats_tr" "ark:ali-to-post ark:train32.lab ark:- |" "$feats_dv" "ark:ali-to-post ark:dev32.lab ark:- |" model.nnet || exit 1
fi

nnet-concat $timit/exp/dnn4_pretrain-dbn_dnn_smbr/final.nnet model.nnet final.nnet || exit 1;

$nn_forward final.nnet $source_tr ark:- | replace-feats ark:- ark:train.ark 0 1.0 || exit 1;
$nn_forward final.nnet $source_dv ark:- | replace-feats ark:- ark:dev.ark   0 1.0 || exit 1;
$nn_forward final.nnet $source_ts ark:- | replace-feats ark:- ark:test.ark  0 1.0 || exit 1;
