#!/bin/zsh

test_align_file=~/Research/kftt-alignments/alignment_$1_$2_$3.txt
test_output_file=~/Research/kftt-alignments/test_output_$1_$2_$3.txt
train_process_output_file=~/Research/kftt-alignments/train_process_$1_$2_$3.txt

for i in {0..4}
do
	test_align_file=~/Research/kftt-alignments/alignment_$1_$2_$i.txt
	test_output_file=~/Research/kftt-alignments/test_output_$1_$2_$i.txt
	train_process_output_file=~/Research/kftt-alignments/train_process_$1_$2_$i.txt
	echo "writing to "$test_align_file","$test_output_file","$train_process_output_file
	~/Research/aligner/aligner -S 2 -H -l $1 -I $2 --history $i --finput ~/Research/kftt-data-1.0/data/tok/kyoto-train.en --einput ~/Research/kftt-data-1.0/data/tok/kyoto-train.ja --etestset ~/Research/kftt-alignments/data/japanese-test.txt --ftestset ~/Research/kftt-alignments/data/english-test.txt --test_align_output_file $test_align_file --train_process_output_file $train_process_output_file --test_output_file $test_output_file
	test_align_file=~/Research/kftt-alignments/alignment_$1_$2_$i-rev.txt
	test_output_file=~/Research/kftt-alignments/test_output_$1_$2_$i-rev.txt
	train_process_output_file=~/Research/kftt-alignments/train_process_$1_$2_$i-rev.txt
	echo "writing to "$test_align_file","$test_output_file","$train_process_output_file
	~/Research/aligner/aligner -S 2 -H -l $1 -I $2 --history $i --finput ~/Research/kftt-data-1.0/data/tok/kyoto-train.en --einput ~/Research/kftt-data-1.0/data/tok/kyoto-train.ja --etestset ~/Research/kftt-alignments/data/japanese-test.txt --ftestset ~/Research/kftt-alignments/data/english-test.txt --test_align_output_file $test_align_file --train_process_output_file $train_process_output_file --test_output_file $test_output_file -r
done
