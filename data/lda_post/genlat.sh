#!/bin/bash

rm -f train.lat test.lat
for i in $(seq 1 20);do gunzip -c ../../timit/exp/dnn4_pretrain-dbn_dnn_smbr/decode_tr_it6/lat.$i.gz >> train.lat; done
for i in $(seq 1 20);do gunzip -c ../../timit/exp/dnn4_pretrain-dbn_dnn_smbr/decode_test_it6/lat.$i.gz >> test.lat; done
