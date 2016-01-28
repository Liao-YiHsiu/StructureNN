#!/bin/bash 

tmpdir=$(mktemp -d)

if [ "$#" -ne 1 ]; then
   echo "Draw loss, accuracy, WER for training procedure."
   echo "Usage: $0 <path to log>"
   echo "eg. $0 data/nnet/1_10_2_400_0.0001_0.16/log"
   echo ""
   exit 1;
fi

log=$1

cat $log | sed 's/%/ /g' > $tmpdir/data

#cat $dir/nnet1* | sed 's/^.*nnet1\.//g' | \
#   sed -e 's/_rejected//g' -e 's/_keep\-lr\-iters\-20//g' \
#   -e 's/_learnrate/ /g' -e 's/_tr/ /g' -e 's/_cv/ /g' \
#   -e 's/_wer/ /g' -e 's/_acc/ /g' -e 's/%/ /g' > $tmpdir/data

gnuplot -persist<<- END
set style data lines;
set title "loss-train";
plot "$tmpdir/data" using 1:3 notitle
pause -1
END

gnuplot -persist<<- END
set style data lines;
set title "loss-test";
plot "$tmpdir/data" using 1:4 notitle
pause -1
END

gnuplot -persist<<- END
set style data lines;
set title "accuracy";
plot "$tmpdir/data" using 1:5 title 'train', "$tmpdir/data" using 1:6 title 'test';
pause -1
END

gnuplot -persist<<- END
set style data lines;
set title "learn-rate";
plot "$tmpdir/data" using 1:2 notitle
pause -1
END

gnuplot -persist<<- END
set style data lines;
set title "wer";
plot "$tmpdir/data" using 1:7 notitle
pause -1
END

rm -rf $tmpdir
