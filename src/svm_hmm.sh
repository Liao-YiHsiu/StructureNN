#!/bin/bash
. ../timit/path.sh

FEAT=../timit/data-fmllr-tri3/train/feats.scp
TEST=../timit/data-fmllr-tri3/test/feats.scp
DIR=data2
CORES=5

#----------------train----------------------
rm -rf $DIR 
mkdir $DIR 
split -l $(( `wc -l < $FEAT` / CORES + 1)) $FEAT $DIR/split

for line in $DIR/split*
do
   mv $line ${line}.scp
done


for line in $DIR/split*.scp
do
   copy-feats scp:$line ark:- | \
   nnet-forward --prior-scale=1.0 --feature-transform=../timit/exp/dnn4_pretrain-dbn_dnn_smbr/final.feature_transform --use-gpu=yes ../timit/exp/dnn4_pretrain-dbn_dnn_smbr/final.nnet ark,s,cs:- ark:- | \
   ./post2phn ../timit/exp/tri3_ali/tree ark,s,cs:- ark:- | \
   ./svm_hmm /corpus/timit/ ../timit/conf/phones.60-48-39.map ../timit/data/lang/phones.txt ark,s,cs:- ${line}.out  
   ../svm_hmm/svm_hmm_learn -c 1000 -e 0.5 ${line}.out ${line}.model &>${line}.log &
   echo "SVM training start..................................."
done

wait

# combine models....
#./combine $DIR/final.model $DIR/split*.model

#----------------test-----------------------
echo "start testing"
   copy-feats scp:$TEST ark:- | \
   nnet-forward --prior-scale=1.0 --feature-transform=../timit/exp/dnn4_pretrain-dbn_dnn_smbr/final.feature_transform --use-gpu=yes ../timit/exp/dnn4_pretrain-dbn_dnn_smbr/final.nnet ark,s,cs:- ark:- | \
   ./post2phn ../timit/exp/tri3_ali/tree ark,s,cs:- ark:- | \
   ./svm_hmm /corpus/timit/ ../timit/conf/phones.60-48-39.map ../timit/data/lang/phones.txt ark,s,cs:- $DIR/test.out  

   for line in $DIR/split*.scp
   do
      ../svm_hmm/svm_hmm_classify $DIR/test.out ${line}.model ${line}.tags &>>${line}.log &
   done

exit 0;
