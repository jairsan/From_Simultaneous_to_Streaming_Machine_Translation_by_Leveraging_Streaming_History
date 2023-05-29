SRC_INPUT_FILE=ACT/ACT_src_reseg
DAL_SCALE=0.85;

RESEGMENTED_H=ACT/ACT_tgt_reseg
ACTION_FILE=ACT/ACT_RW
lat=$(python3 eval_streaming_latency.py $SRC_INPUT_FILE $RESEGMENTED_H $ACTION_FILE $DAL_SCALE)

echo $bleu $lat

