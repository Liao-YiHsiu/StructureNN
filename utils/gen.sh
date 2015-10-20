#!/bin/bash -ex

# auto genarate commonly used files.

#timit="/home/loach/Research/timit"
DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
[ -d $DIR/../data ] || mkdir -p $DIR/../data
cd $DIR/../data

source $DIR/../path

source_tr="scp,s,cs:$timit/data-fmllr-tri3/train/feats.scp"
source_ts="scp,s,cs:$timit/data-fmllr-tri3/test/feats.scp"
source_dv="scp,s,cs:$timit/data-fmllr-tri3/dev/feats.scp"

files="train.lab dev.lab test.lab train32.lab dev32.lab test32.lab train.lat test.lat dev.lat "

rm -f $files

gen-lab $timit_corpus $timit/conf/phones.60-48-39.map $timit/data/lang/words.txt $source_tr ark:- | \
      tee train.lab | uchar-to-int32 ark:- ark:train32.lab || exit 1;
gen-lab $timit_corpus $timit/conf/phones.60-48-39.map $timit/data/lang/words.txt $source_dv ark:- | \
      tee dev.lab   | uchar-to-int32 ark:- ark:dev32.lab   || exit 1;
gen-lab $timit_corpus $timit/conf/phones.60-48-39.map $timit/data/lang/words.txt $source_ts ark:- | \
      tee test.lab  | uchar-to-int32 ark:- ark:test32.lab  || exit 1;

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

# create directory.
for script in $DIR/gen_*.sh;
do
   base=$(basename $script)
   tmp=${base#gen_*_}
   dir=${tmp%.sh}
   rm -rf $dir
   mkdir $dir

# link files
   for file in $files;
   do
      rm -f $dir/$file
      ln -sf ../$file $dir/$file
   done
done

# generate features.
for file in $DIR/gen_*.sh;
do
   $file
done
