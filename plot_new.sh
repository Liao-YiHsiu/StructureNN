#!/bin/bash

max=-1
log=""

for file in data/*/*/log ;
do
   stamp=$(stat -c %Y $file)
   if [[ $stamp -gt $max ]];then
      max=$stamp
      log=$file
   fi
done

echo $log
./utils/plot.sh $log
