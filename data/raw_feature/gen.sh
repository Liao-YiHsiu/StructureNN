#!/bin/bash
. ../path.sh

copy-feats $source_tr ark:train.ark || exit 1;
copy-feats $source_dv ark:dev.ark   || exit 1;
copy-feats $source_ts ark:test.ark  || exit 1;

