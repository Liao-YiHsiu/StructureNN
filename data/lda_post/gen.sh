#!/bin/bash

dim=48

. ../path.sh

echo "Estimating LDA statics"
lab-lda "$feats_tr" ark:train.lab ldaac || exit 1;
est-lda --dim=$dim --write-full-matrix=lda.full.mat lda.mat ldaac || exit 1;

transf-to-nnet lda.mat lda.nnet

nnet-concat $timit/exp/dnn4_pretrain-dbn_dnn_smbr/final.nnet lda.nnet final.nnet

$nn_forward final.nnet $source_tr ark:train.ark
$nn_forward final.nnet $source_dv ark:dev.ark
$nn_forward final.nnet $source_ts ark:test.ark
