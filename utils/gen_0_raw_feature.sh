#!/bin/bash

DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
source $DIR/genpath.sh
cd $DIR/../data/raw_feature


copy-feats $source_tr ark:train.ark || exit 1;
copy-feats $source_dv ark:dev.ark   || exit 1;
copy-feats $source_ts ark:test.ark  || exit 1;

cat > nnet1 <<EOF
<Nnet>
<Splice> 40 40
[ 0 ]
</Nnet>
EOF
