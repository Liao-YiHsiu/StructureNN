#!/bin/bash -ex
source path

tmpdir=$(mktemp -d)
trap 'rm -rf $tmpdir ' EXIT

if [ "$#" -ne 1 ]; then
   echo "Visualize score-path zip results"
   echo "Usage: $0 <score-path-zipfile>"
   echo "eg. $0 data/nnet/1_10_2_400_0.0001_0.16.data.tag.gz"
   echo ""
   exit 1;
fi

zipfile=$1
dir=$(dirname $zipfile)

pred_score="ark:gunzip -c $zipfile |"
real_score="ark:gunzip -c $tmpdir/real.tgz.gz |"

   # PER or FER of a sentence.
   score-oracle ark:$dir/../test.lab "$pred_score" "ark:| gzip -c > $tmpdir/real.tgz.gz" 

   # generating the points.
   score-path-point "$pred_score" "$real_score" > $tmpdir/out

for key in $(cut -f1 $tmpdir/out | uniq)
do
   grep "$key" $tmpdir/out > $tmpdir/$key
   cat >> $tmpdir/script <<END
   set title "$key";
   plot "$tmpdir/$key" using 2:3 pt 7 ps 2 notitle
   pause mouse any "press Enter to exit"
   if (MOUSE_KEY == 13) exit
END
done

gnuplot < $tmpdir/script
rm -rf $tmpdir
