#!/bin/bash
echo "$0 $@"  # Print the command line for logging

. parse_options.sh || exit 1;

if [ "$#" -ne 2 ]; then
   echo "Calculate the Error Rate on the model"
   echo "Usage: $0 <ref-rspecifier> <path-score-rspecifier>"
   echo "eg. $0 ark:test.lab ark:test.tag"
   echo ""
   echo "dir-> $files"
   exit 1;
fi
lab=$1
path=$2

   echo "Calculating Error rate."

   path-fer "$1" "ark:split-path-score \"$2\" ark:/dev/null ark:- |" 

   compute-wer "ark:trim-path \"$1\" ark:- |" "ark:split-path-score \"$2\" ark:/dev/null ark:- | trim-path ark:- ark:- |" 

   echo "Calculating Error rate.(39)" 

   path-fer "ark:trans.sh \"$1\" ark:- |" "ark:split-path-score \"$2\" ark:/dev/null ark:- | trans.sh ark:- ark:- |" 

   compute-wer "ark:trim-path \"$1\" ark:- | trans.sh ark:- ark:- |" "ark:split-path-score \"$2\" ark:/dev/null ark:- | trim-path ark:- ark:- | trans.sh ark:- ark:- |" 
