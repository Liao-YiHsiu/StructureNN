#!/bin/bash

# auto genarate commonly used files.

timit="/home/loach/Research/timit"

source_tr="scp,s,cs:$timit/data-fmllr-tri3/train/feats.scp"
source_ts="scp,s,cs:$timit/data-fmllr-tri3/test/feats.scp"
source_dv="scp,s,cs:$timit/data-fmllr-tri3/dev/feats.scp"

files="train.lab dev.lab test.lab train.lat test.lat dev.lat"

rm -f $files

gen-lab /corpus/timit $timit/conf/phones.60-48-39.map $timit/data/lang/words.txt $source_tr ark:train.lab || exit 1;
gen-lab /corpus/timit $timit/conf/phones.60-48-39.map $timit/data/lang/words.txt $source_dv ark:dev.lab || exit 1;
gen-lab /corpus/timit $timit/conf/phones.60-48-39.map $timit/data/lang/words.txt $source_ts ark:test.lab || exit 1;


rm -f train.lat test.lat dev.lat
for i in $(seq 1 20);do gunzip -c $timit/exp/dnn4_pretrain-dbn_dnn_smbr/decode_tr_it6/lat.$i.gz >> train.lat; done
for i in $(seq 1 20);do gunzip -c $timit/exp/dnn4_pretrain-dbn_dnn_smbr/decode_test_it6/lat.$i.gz >> test.lat; done
for i in $(seq 1 20);do gunzip -c $timit/exp/dnn4_pretrain-dbn_dnn_smbr/decode_dev_it6/lat.$i.gz >> dev.lat; done

# link files

for dir in *;
do
   if [ -d "$dir" ];
   then
      for file in $files;
      do
         rm -f $dir/$file
         ln -sf ../$file $dir/$file
      done
   fi
done
