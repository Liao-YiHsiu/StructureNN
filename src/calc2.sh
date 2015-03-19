#!/bin/bash
echo "$0 $@"  # Print the command line for logging

. parse_options.sh || exit 1;

if [ "$#" -ne 2 ]; then
   echo "Calculate the Error Rate on the model"
   echo "Usage: $0 <ref-rspecifier> <lab-rspecifier>"
   echo "eg. $0 ark:test.lab ark:decode.lab"
   echo ""
   echo "dir-> $files"
   exit 1;
fi
ref=$1
lab=$2

   echo "Calculating Error rate."

   path-fer "$ref" "$lab"

   compute-wer "ark:trim-path \"$ref\" ark:- |" "ark:trim-path \"$lab\" ark:- |" 

   echo "Calculating Error rate.(39)" 

   path-fer "ark:trans.sh \"$ref\" ark:- |" "ark:trans.sh \"$lab\" ark:- |" 

   compute-wer "ark:trim-path \"$ref\" ark:- | trans.sh ark:- ark:- |" "ark:trim-path \"$lab\" ark:- | trans.sh ark:- ark:- |" 
