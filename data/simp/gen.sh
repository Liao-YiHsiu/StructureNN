#!/bin/bash

DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
cd $DIR
PATH=$PATH:$DIR/../../src

source ../path.sh

rm train.ark test.ark train.lab test.lab data.out test.out dev.ark dev.lab dev.out

sed raw/train.ark -e "s/[a-zA-Z]*-dr[0-9]-//g" -e "s/-\([a-zA-Z]\)/_\1/g" -e "s/s\([a-zA-Z]\)[0]*\([0-9]\+\)/s\1\2/g" > train.ark || exit 1;
sed raw/dev.ark -e "s/[a-zA-Z]*-dr[0-9]-//g" -e "s/-\([a-zA-Z]\)/_\1/g" -e "s/s\([a-zA-Z]\)[0]*\([0-9]\+\)/s\1\2/g" > dev.ark  || exit 1;
sed raw/test.ark -e "s/[a-zA-Z]*-dr[0-9]-//g" -e "s/-\([a-zA-Z]\)/_\1/g" -e "s/s\([a-zA-Z]\)[0]*\([0-9]\+\)/s\1\2/g" > test.ark  || exit 1;

gen-lab $timit_corpus $timit/conf/phones.60-48-39.map $timit/data/lang/words.txt ark:train.ark ark:train.lab  || exit 1;
gen-lab $timit_corpus $timit/conf/phones.60-48-39.map $timit/data/lang/words.txt ark:dev.ark ark:dev.lab  || exit 1;
gen-lab $timit_corpus $timit/conf/phones.60-48-39.map $timit/data/lang/words.txt ark:test.ark ark:test.lab  || exit 1;

con-svm ark:train.lab ark:train.ark data.out || exit 1;
con-svm ark:dev.lab ark:dev.ark dev.out ||exit 1;
con-svm ark:test.lab ark:test.ark test.out ||exit 1;
