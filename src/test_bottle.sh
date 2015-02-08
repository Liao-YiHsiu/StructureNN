#!/bin/bash

. ../timit/path.sh

FEAT=data-fmllr-tri3/train/feats.scp
TEST=../timit/data-fmllr-tri3/test/feats.scp

./bottleneck.sh --bn-dim 48 --train-opts "--max-iters 100 --min-iters 20" "ark:nnet-forward --prior-scale=1.0 --feature-transform=exp/dnn4_pretrain-dbn_dnn_smbr/final.feature_transform --use-gpu=yes exp/dnn4_pretrain-dbn_dnn_smbr/final.nnet scp,s,cs:$FEAT  ark:- |" model.nnet

