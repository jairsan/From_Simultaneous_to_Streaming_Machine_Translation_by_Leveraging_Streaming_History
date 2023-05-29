SRC_INPUT_FILE=norm.iwslt17.tst2010.de
for DAL_SCALE in 0.85;
do
	for segmentation in segmented_IWSLT20_rnn_11_0 segmented_IWSLT20_rnn_12_1 segmented_IWSLT20_rnn_13_2 segmented_IWSLT20_rnn_14_3 segmented_IWSLT20_rnn_15_4;
	do
		echo $segmentation
		for k in 1 2 4 8 16 32;
		do
			RESEGMENTED_H=reproducible_$DAL_SCALE/$segmentation.$k.reseg_h
			ACTION_FILE=reproducible_$DAL_SCALE/$segmentation.$k.RW
			lat=$(python3 eval_streaming_latency.py $SRC_INPUT_FILE $RESEGMENTED_H $ACTION_FILE $DAL_SCALE)
			bleu=$(cat $RESEGMENTED_H | ./detokenizer.perl -l en | sacrebleu -l de-en iwslt17.tst2010.en | cut -f 3 -d " ")
			echo $bleu $lat
		done
	done
done
	
