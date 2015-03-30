#!/bin/bash

echo "$0 $@"  # Print the command line for logging
command_line="$0 $@"

tmpdir=$(mktemp -d)
lattice_N=2000
acwt=0.16
N=10

. parse_options.sh || exit 1;

if [ "$#" -ne 2 ]; then
   echo "Combine score path (from lattice) and SDNN model to reweight."
   echo "Usage: $0 <dir> <SDNN-model>"
   echo "eg. $0 data/simp data/simp/data_nn.model_1000_1_100"
   echo ""
   exit 1;
fi

dir=$1
model=$2

log=${model}.rescore.log

lattice_N_times=$((lattice_N))

dev_lattice_path_gz="$dir/dev.lab_${lattice_N_times}_${acwt}.gz"
test_lattice_path_gz="$dir/test.lab_${lattice_N_times}_${acwt}.gz"

dev_lattice_path="ark:gunzip -c $dev_lattice_path_gz |"
test_lattice_path="ark:gunzip -c $test_lattice_path_gz |"

dev_lab="ark:$dir/dev.lab"
test_lab="ark:$dir/test.lab"

files="$model $dev_lattice_path_gz $test_lattice_gz $dir/test.ark $dir/dev.ark $dir/dev.lab $dir/test.lab"

   echo $command_line \
      2>&1 | tee $log ; ( exit ${PIPESTATUS[0]} ) || exit 1;

   #check file existence.
   for file in $files;
   do
       if [ ! -f $file ]; then 
          echo "File '$file' not found." 
          exit 1 
       fi
   done

   test_model=${model}.tag
   [ -f $test_model ] || snnet-score ark:$dir/test.ark "$test_lattice_path" $model ark:$test_model\
      2>&1 | tee -a $log ; ( exit ${PIPESTATUS[0]} ) || exit 1;

   dev_model=${model}.tag_dev
   [ -f $dev_model ] || snnet-score ark:$dir/dev.ark "$dev_lattice_path" $model ark:$dev_model\
      2>&1 | tee -a $log ; ( exit ${PIPESTATUS[0]} ) || exit 1;
   

   #prepare dev data
   weight-score-path ark:- -1 "$dev_lattice_path" | \
      normalize-score-path --log-domain=true ark:- ark:- | \
      cmvn-score-path ark:- ark:$tmpdir/lat \
      2>&1 | tee -a $log ; ( exit ${PIPESTATUS[0]} ) || exit 1;

   log-score-path "ark:$dev_model" ark:- | \
      normalize-score-path --log-domain=true ark:- ark:- | \
      cmvn-score-path ark:- ark:$tmpdir/lab \
      2>&1 | tee -a $log ; ( exit ${PIPESTATUS[0]} ) || exit 1;

   pi=$(echo "scale=10; 4*a(1)" | bc -l)

   # use dev set to tune the weight.
   for ((n = 0; n <= $N; n++))
   do
      theta=$(echo "$pi*$n/2/$N" | bc -l)
      w=$(echo "s($theta)/c($theta)" | bc -l)
      tmp_log=$tmpdir/log.$n

      compute-wer "ark:trim-path \"$dev_lab\" ark:- | trans.sh ark:- ark:- |" "ark:weight-score-path ark:- 1 ark:$tmpdir/lat $w ark:$tmpdir/lab | best-score-path ark:- ark:- |split-score-path ark:- ark:/dev/null ark:- | trim-path ark:- ark:- | trans.sh ark:- ark:- |" 2>&1 >$tmp_log &
   done

   for job in `jobs -p`
   do
      wait $job || exit 1; 
   done
   
   min=100
   best_w=-1

   for ((n = 0; n <= $N; n++))
   do
      theta=$(echo "$pi*$n/2/$N" | bc -l)
      w=$(echo "s($theta)/c($theta)" | bc -l)
      tmp_log=$tmpdir/log.$n

      wer=$(cat $tmp_log | grep 'WER' | tail -n 1 | awk '{ print $2; }')
      echo "w = $w , WER = $wer "\
         2>&1 | tee -a $log ; ( exit ${PIPESTATUS[0]} ) || exit 1;

      if [ 1 == $(echo "$wer < $min" | bc -l) ]; then
         min=$wer
         best_w=$w
      fi
      
   done

   echo "BEST WER is $min with w = $best_w" \
      2>&1 | tee -a $log ; ( exit ${PIPESTATUS[0]} ) || exit 1;

   weight-score-path ark:- -1 "$test_lattice_path" | \
      normalize-score-path --log-domain=true ark:- ark:- | \
      cmvn-score-path ark:- ark:$tmpdir/lat \
      2>&1 | tee -a $log ; ( exit ${PIPESTATUS[0]} ) || exit 1;

   log-score-path "ark:$test_model" ark:- | \
      normalize-score-path --log-domain=true ark:- ark:- | \
      cmvn-score-path ark:- ark:$tmpdir/lab \
      2>&1 | tee -a $log ; ( exit ${PIPESTATUS[0]} ) || exit 1;

   compute-wer "ark:trim-path \"$test_lab\" ark:- | trans.sh ark:- ark:- |" "ark:weight-score-path ark:- 1 ark:$tmpdir/lat $best_w ark:$tmpdir/lab | best-score-path ark:- ark:- |split-score-path ark:- ark:/dev/null ark:- | trim-path ark:- ark:- | trans.sh ark:- ark:- |" \
      2>&1 | tee -a $log ; ( exit ${PIPESTATUS[0]} ) || exit 1;

#   rm -rf $tmpdir

exit 0;
