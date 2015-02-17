#!/bin/bash

timit="/home/loach/Research/timit"

source_tr="scp,s,cs:$timit/data-fmllr-tri3/train/feats.scp"
source_dv="scp,s,cs:$timit/data-fmllr-tri3/dev/feats.scp"
source_ts="scp,s,cs:$timit/data-fmllr-tri3/test/feats.scp"

nn_forward="nnet-forward --prior-scale=1.0 --feature-transform=$timit/exp/dnn4_pretrain-dbn_dnn_smbr/final.feature_transform --use-gpu=yes"

feats_tr="ark:$nn_forward $timit/exp/dnn4_pretrain-dbn_dnn_smbr/final.nnet $source_tr ark:- |"
feats_dv="ark:$nn_forward $timit/exp/dnn4_pretrain-dbn_dnn_smbr/final.nnet $source_dv ark:- |"
feats_ts="ark:$nn_forward $timit/exp/dnn4_pretrain-dbn_dnn_smbr/final.nnet $source_ts ark:- |"
