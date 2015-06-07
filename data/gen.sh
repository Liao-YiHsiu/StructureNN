#!/bin/bash

# auto genarate commonly used files.

#timit="/home/loach/Research/timit"
DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
cd $DIR

source ../path

source_tr="scp,s,cs:$timit/data-fmllr-tri3/train/feats.scp"
source_ts="scp,s,cs:$timit/data-fmllr-tri3/test/feats.scp"
source_dv="scp,s,cs:$timit/data-fmllr-tri3/dev/feats.scp"

files="train.lab dev.lab test.lab train.lat test.lat dev.lat"

rm -f $files

gen-lab $timit_corpus $timit/conf/phones.60-48-39.map $timit/data/lang/words.txt $source_tr ark:train.lab || exit 1;
gen-lab $timit_corpus $timit/conf/phones.60-48-39.map $timit/data/lang/words.txt $source_dv ark:dev.lab || exit 1;
gen-lab $timit_corpus $timit/conf/phones.60-48-39.map $timit/data/lang/words.txt $source_ts ark:test.lab || exit 1;

#[ -e $timit/exp/tri3/decode_tr ] || (
#		cd $timit
#		steps/decode_fmllr.sh --nj 5 --cmd run.pl \
#		exp/tri3/graph data/train exp/tri3/decode_tr
#)
#
#[ -e $timit/exp/dnn4_pretrain-dbn_dnn_smbr/decode_tr_it6 ] || (
#		decode_extra_opts=(--num-threads 6 --parallel-opts "-pe smp 6 -l mem_free=4G,ram_free=4.0G")
#		cd $timit 
#		steps/nnet2/decode.sh --cmd run.pl --nj 5 "${decode_extra_opts[@]}" \
#		--transform-dir exp/tri3/decode_tr exp/tri3/graph data/train \
#		exp/tri4_nnet/decode_tr 
#		)

[ -e $timit/exp/dnn4_pretrain-dbn_dnn_smbr/decode_tr_it6 ] || (
		dir=exp/dnn4_pretrain-dbn_dnn_smbr
		srcdir=exp/dnn4_pretrain-dbn_dnn
		data_fmllr=data-fmllr-tri3
		gmmdir=exp/tri3
		acwt=0.2

		cd $timit
		steps/nnet/decode.sh --nj 20 --cmd run.pl \
		--nnet $dir/6.nnet --acwt $acwt \
		$gmmdir/graph $data_fmllr/train $dir/decode_tr_it6 
		)

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
