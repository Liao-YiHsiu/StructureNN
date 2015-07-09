#!/bin/bash

DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
source $DIR/path.sh
cd $DIR/../data

dirname=nnet_raw

[ -d $dirname ] || mkdir $dirname 

cd $dirname
[ -e train.ark ] || ln -sf ../raw_feature/train.ark train.ark
[ -e dev.ark   ] || ln -sf ../raw_feature/dev.ark   dev.ark
[ -e test.ark  ] || ln -sf ../raw_feature/test.ark  test.ark
[ -e nnet1     ] || ln -sf ../nn_post/final.nnet    nnet1

for file in $(ls ..);
do
   [ -f ../$file ] && [ ! -e $file ] && ln -sf ../$file $file
done
