#!/bin/bash

max=-1
thres=6000

for file in data/*/*/log ;
do
   stamp=$(stat -c %Y $file)
   if [[ $stamp -gt $max ]];then
      max=$stamp
   fi
done

log_list=""
max=$((max - thres))

for file in data/*/*/log ;
do
   stamp=$(stat -c %Y $file)
   if [[ $stamp -gt $max ]];then
      log_list="$log_list $file"
   fi
done

for file in $log_list;
do
   echo $file
   ./utils/plot.sh $file
   read "pause"
done
