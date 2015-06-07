#!/bin/bash

DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
source $DIR/../path

from=48
to=39

# transcribe from 39 to 48

. parse_options.sh || exit 1;

if [ "$#" -ne 2 ]; then
   echo "Transcribe labels from 48 to 39 phones"
   echo "Usage: $0 <rspecifier> <wspecifier> "
   echo ""
   exit 1;
fi

input=$1
output=$2

copy-int-vector "$input"  ark,t:- | \
      $timit/utils/int2sym.pl -f 2- $timit/data/lang/words.txt - | \
      $timit/local/timit_norm_trans.pl -i - -m $timit/conf/phones.60-48-39.map -from $from -to $to | \
      $timit/utils/sym2int.pl -f 2- $timit/data/lang/words.txt - |\
      copy-int-vector ark:- "$output"
