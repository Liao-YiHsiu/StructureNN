#!/bin/bash

. ../path.sh

[ -f model.nnet ] || ./bottleneck.sh --bn-dim 48 --max-iters 100 --min-iters 10 "$feats_tr" "$feats_dv" model.nnet || exit 1;

nnet-concat $timit/exp/dnn4_pretrain-dbn_dnn_smbr/final.nnet model.nnet final.nnet || exit 1;

$nn_forward final.nnet $source_tr ark:train.ark || exit 1;
$nn_forward final.nnet $source_dv ark:dev.ark || exit 1;
$nn_forward final.nnet $source_ts ark:test.ark || exit 1;
