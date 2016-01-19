#!/bin/bash

DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
source $DIR/genpath.sh
cd $DIR/../data/nnetraw

[ -e train.ark ]   || ln -sf ../raw_feature/train.ark train.ark
[ -e dev.ark   ]   || ln -sf ../raw_feature/dev.ark   dev.ark
[ -e test.ark  ]   || ln -sf ../raw_feature/test.ark  test.ark
[ -e nnet1     ]   || nnet-pop --num=2 $timit/exp/dnn4_pretrain-dbn_dnn_smbr/final.nnet nnet1
[ -e transf.nnet ] || ln -sf $timit/exp/dnn4_pretrain-dbn_dnn_smbr/final.feature_transform transf.nnet
