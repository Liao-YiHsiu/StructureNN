#!/bin/bash
. ../timit/path.sh

FEAT=../timit/data-fmllr-tri3/train/feats.scp
CORES=7

rm -rf data
mkdir data
split -l $(( `wc -l < $FEAT` / CORES )) $FEAT data/split

for line in data/split*
do
   mv $line ${line}.scp
done


for line in data/split*.scp
do
   copy-feats scp:$line ark:- | \
   nnet-forward --no-softmax=true --prior-scale=1.0 --feature-transform=../timit/exp/dnn4_pretrain-dbn_dnn_smbr/final.feature_transform --class-frame-counts=../timit/exp/dnn4_pretrain-dbn_dnn_smbr/ali_train_pdf.counts --use-gpu=no ../timit/exp/dnn4_pretrain-dbn_dnn_smbr/final.nnet ark,s,cs:- ark:- | \
   ./svm_hmm /corpus/timit/ ../timit/conf/phones.60-48-39.map ark,s,cs:- ${line}.out 
   ../svm_hmm/svm_hmm_learn -c 5 -e 0.5 ${line}.out ${line}.model &>${line}.log &
   echo "SVM training start..................................."
done

wait

exit 0;
