#!/bin/bash
. ../timit/path.sh

TRAIN=../timit/data-fmllr-tri3/train/feats.scp
TEST=../timit/data-fmllr-tri3/test/feats.scp
DIR=data3

[ -d $DIR ] || mkdir $DIR

   ./svm_hmm /corpus/timit/ ../timit/conf/phones.60-48-39.map ../timit/data/lang/phones.txt scp:$TRAIN $DIR/train.out  
   ../svm_hmm/svm_hmm_learn -c 1000 -e 0.5 $DIR/train.out $DIR/train.model &>$DIR/train.log

   ./svm_hmm /corpus/timit/ ../timit/conf/phones.60-48-39.map ../timit/data/lang/phones.txt scp:$TEST $DIR/test.out  
   ../svm_hmm/svm_hmm_classify $DIR/test.out $DIR/train.model $DIR/test.tags &>$DIR/test.log
