#!/bin/zsh
base=~/Research/kftt-alignments

for i in {0..4}
do
$base/bin/measure-alignment-error.pl $base/data/align-test.txt $base/symmetric-$1_$2_$i.txt > $base/symmetric-$1_$2_$i.acc
$base/bin/measure-alignment-error.pl $base/data/align-test.txt $base/alignment_$1_$2_$i.txt > $base/alignment_$1_$2_$i.acc
done
