function eval_setup {
	catchup=$1
	waitk=$2
	folder=STR-MT/StreamTranslation_ende_txt.Transformer_BIG_waitk_doc_partial_bidirection_60history_stream_strict_mode_125total_length_ups5_SPM_CC_filtered.norm.doc.MuST-C.test.results_MuST-C_ADAM_wait"$waitk"_catchup"$catchup"_beam4
	bleu=$(cat $folder/text | sed -r 's/\@\@ //g' | sed -r 's#\[DOC\]##g' | sed -r 's#<unk>##g' | sed -r 's#\[SEP\]##g' | sed -r 's#\[BRK\]##g' | sed -r 's#\[CONT\]##g' | ./detruecase.perl | sacrebleu --score-only -l en-de MuST-C.test.de)
	ans=$(python3 -W ignore eval_latency.py --input $folder/sentence_valid_delays | cut -f2 -d " ")
	echo $bleu $ans
}

eval_setup 1.0 2
eval_setup 1.0 3
eval_setup 0.8 2
eval_setup 0.8 3
eval_setup 0.8 4
eval_setup 0.8 5
eval_setup 0.8 6
eval_setup 0.8 7
eval_setup 0.8 10

