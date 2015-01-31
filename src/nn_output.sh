#!/bin/bash
. ../timit/path.sh
nnet-forward --no-softmax=true --prior-scale=1.0 --feature-transform=../timit/exp/dnn4_pretrain-dbn_dnn_smbr/final.feature_transform --class-frame-counts=../timit/exp/dnn4_pretrain-dbn_dnn_smbr/ali_train_pdf.counts --use-gpu=no ../timit/exp/dnn4_pretrain-dbn_dnn_smbr/1.nnet 'ark,s,cs:copy-feats scp:../timit/data-fmllr-tri3/test/split20/1/feats.scp ark:- |' ark:- | \
./str_format /corpus/timit/ ../timit/conf/phones.60-48-39.map ark:- ark,t:-|less

exit 0;
nnet-forward --no-softmax=true --prior-scale=1.0 --feature-transform=exp/dnn4_pretrain-dbn_dnn_smbr/final.feature_transform --class-frame-counts=exp/dnn4_pretrain-dbn_dnn_smbr/ali_train_pdf.counts --use-gpu=no exp/dnn4_pretrain-dbn_dnn_smbr/1.nnet 'ark,s,cs:copy-feats scp:data-fmllr-tri3/test/split20/1/feats.scp ark:- |' ark,t:tmp.ark
