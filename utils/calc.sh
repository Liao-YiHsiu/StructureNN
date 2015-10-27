#!/bin/bash -ex
echo "$0 $@"  # Print the command line for logging

DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
source $DIR/../path

. parse_options.sh || exit 1;

if [ "$#" -ne 2 ]; then
   echo "Calculate the Error Rate on the model (uchar version)"
   echo "Usage: $0 <ref-rspecifier> <score-path-rspecifier>"
   echo "eg. $0 ark:test.lab ark:test.tag"
   echo ""
   echo "dir-> $files"
   exit 1;
fi
lab=$1
path=$2

   echo "Calculating Error rate."

   path-fer "$1" "ark:split-score-path \"$2\" ark:/dev/null ark:- |" 

   compute-wer "ark:trim-path \"$1\" ark,t:- |" \
   "ark:split-score-path \"$2\" ark:/dev/null ark:- | trim-path ark:- ark,t:- |" 

   echo "Calculating Error rate.(39)" 

   path-fer "ark:trans.sh \"$1\" ark:- |" \
      "ark:split-score-path \"$2\" ark:/dev/null ark:- | trans.sh ark:- ark:- |" 

   compute-wer "ark:trans.sh \"$1\" ark:- | trim-path ark:- ark,t:- |" \
      "ark:split-score-path \"$2\" ark:/dev/null ark:- | trans.sh ark:- ark:- | trim-path ark:- ark,t:- |" 
