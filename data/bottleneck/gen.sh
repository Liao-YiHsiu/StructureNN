#!/bin/bash

timit="../../timit"

source_tr="scp,s,cs:$timit/data-fmllr-tri3/train/feats.scp"
source_ts="scp,s,cs:$timit/data-fmllr-tri3/test/feats.scp"

nn_forward="nnet-forward --prior-scale=1.0 --feature-transform=$timit/exp/dnn4_pretrain-dbn_dnn_smbr/final.feature_transform --use-gpu=yes"

feats_tr="ark:$nn_forward $timit/exp/dnn4_pretrain-dbn_dnn_smbr/final.nnet $source_tr ark:- |"
feats_ts="ark:$nn_forward $timit/exp/dnn4_pretrain-dbn_dnn_smbr/final.nnet $source_ts ark:- |"

[ -f model.nnet ] || ./bottleneck.sh --bn-dim 48 --max-iters 100 --min-iters 10 "$feats_tr" model.nnet || exit 1;

gen-lab /corpus/timit $timit/conf/phones.60-48-39.map $timit/data/lang/words.txt $source_tr ark:train.lab || exit 1;
gen-lab /corpus/timit $timit/conf/phones.60-48-39.map $timit/data/lang/words.txt $source_ts ark:test.lab || exit 1;

nnet-concat $timit/exp/dnn4_pretrain-dbn_dnn_smbr/final.nnet model.nnet final.nnet || exit 1;

$nn_forward final.nnet $source_tr ark:train.ark || exit 1;
$nn_forward final.nnet $source_ts ark:test.ark || exit 1;
