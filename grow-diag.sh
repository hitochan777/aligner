#!/bin/zsh

for i in {0..4}
do
~/Research/cdec/utils/atools -i ~/Research/kftt-alignments/alignment_$1_$2_$i.txt -j ~/Research/kftt-alignments/alignment_$1_$2_$i-rev.txt -c grow-diag-final-and > ~/Research/kftt-alignments/symmetric-$1_$2_$i.txt
done
