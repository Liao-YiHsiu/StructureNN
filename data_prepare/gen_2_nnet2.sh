#!/bin/bash

DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
source $DIR/path.sh
cd $DIR/../data/nnet2

[ -e train.ark ]   || ln -sf ../raw_feature/train.ark train.ark
[ -e dev.ark   ]   || ln -sf ../raw_feature/dev.ark   dev.ark
[ -e test.ark  ]   || ln -sf ../raw_feature/test.ark  test.ark
[ -e nnet1     ]   || ln -sf ../nn_post2/final.nnet    nnet1
[ -e transf.nnet ] || ln -sf $timit/exp/dnn4_pretrain-dbn_dnn_smbr/final.feature_transform transf.nnet

for file in $(ls ..);
do
   [ -f ../$file ] && [ ! -e $file ] && ln -sf ../$file $file
done
