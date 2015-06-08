#!/bin/bash

DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
source $DIR/path.sh
cd $DIR/../data/raw_feature


copy-feats $source_tr ark:train.ark || exit 1;
copy-feats $source_dv ark:dev.ark   || exit 1;
copy-feats $source_ts ark:test.ark  || exit 1;

