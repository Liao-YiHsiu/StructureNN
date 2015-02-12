#!/bin/bash

timit="../../timit"
source_tr="scp,s,cs:$timit/data-fmllr-tri3/train/feats.scp"
source_ts="scp,s,cs:$timit/data-fmllr-tri3/test/feats.scp"

nn_forward="nnet-forward --prior-scale=1.0 --feature-transform=$timit/exp/dnn4_pretrain-dbn_dnn_smbr/final.feature_transform --use-gpu=no"

feats_tr="ark:$nn_forward $timit/exp/dnn4_pretrain-dbn_dnn_smbr/final.nnet $source_tr ark:- |"
feats_ts="ark:$nn_forward $timit/exp/dnn4_pretrain-dbn_dnn_smbr/final.nnet $source_ts ark:- |"

gen-lab /corpus/timit $timit/conf/phones.60-48-39.map $timit/data/lang/words.txt $source_tr ark:train.lab || exit 1;
gen-lab /corpus/timit $timit/conf/phones.60-48-39.map $timit/data/lang/words.txt $source_ts ark:test.lab || exit 1;

echo "Estimating LDA statics"
lab-lda "$feats_tr" ark:train.lab ldaac || exit 1;
est-lda --dim=48 --write-full-matrix=lda.full.mat lda.mat ldaac || exit 1;

transf-to-nnet lda.mat lda.nnet

nnet-concat $timit/exp/dnn4_pretrain-dbn_dnn_smbr/final.nnet lda.nnet final.nnet

$nn_forward final.nnet $source_tr ark:train.ark
$nn_forward final.nnet $source_ts ark:test.ark
