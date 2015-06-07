#!/bin/bash

DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
cd $DIR

source ../path.sh

copy-feats $source_tr ark:train.ark || exit 1;
copy-feats $source_dv ark:dev.ark   || exit 1;
copy-feats $source_ts ark:test.ark  || exit 1;

