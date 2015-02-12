#!/bin/bash
timit="../../timit"

copy-feats scp:$timit/data-fmllr-tri3/train/feats.scp ark:train.scp || exit 1;
copy-feats scp:$timit/data-fmllr-tri3/test/feats.scp ark:test.scp || exit 1;
gen-lab /corpus/timit $timit/conf/phones.60-48-39.map $timit/data/lang/words.txt ark:train.ark ark:train.lab || exit 1;
gen-lab /corpus/timit $timit/conf/phones.60-48-39.map $timit/data/lang/words.txt ark:test.ark ark:test.lab || exit 1;
