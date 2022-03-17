set -e 

SPIDER_DIR=/vault/spider
PREDS_IN_DIR=~/SpeakQL/SpeakQL/Allennlp_models/outputs
PREDS_OUT_DIR=~/SpeakQL/SpeakQL/Allennlp_models/outputs/uskg-mixed-test-save

# SPIDER_DIR=/Users/mac/Desktop/syt/Deep-Learning/Dataset/spider
# PREDS_IN_DIR=/Users/mac/Desktop/syt/Deep-Learning/Projects-M/SpeakQL/SpeakQL/Allennlp_models/outputs
# PREDS_OUT_DIR=/Users/mac/Desktop/syt/Deep-Learning/Projects-M/SpeakQL/SpeakQL/Allennlp_models/outputs/uskg-mixed-test-save

mkdir -p $PREDS_OUT_DIR

python uskg_infer.py \
	-cfg Salesforce/T5_base_prefix_spider_with_cell_value.cfg \
	-model_name /vault/uskg/output/A-T5_base_prefix_spider_with_cell_value-asr_mixed/checkpoint-66500 \
	-test_dataset_path $SPIDER_DIR/my/dev/test_rewriter+phonemes.json \
	-orig_dev_path $SPIDER_DIR/dev.json \
	-db_path $SPIDER_DIR/database \
	-eval_vers 2.12.1.0t-2.29.0.0i 2.12.1.1t-2.29.0.1i 2.12.1.2t-2.29.0.2i 2.12.1.3t-2.29.0.3i \
	-eval_in_dir $PREDS_IN_DIR \
	-eval_out_dir $PREDS_OUT_DIR

# regular TTS
# -test_dataset_path $SPIDER_DIR/my/dev/test_rewriter+phonemes.json \
# -orig_dev_path $SPIDER_DIR/dev.json \

# human test
# -test_dataset_path $SPIDER_DIR/my/dev/human_test/human_test_yshao_rewriter.json \
# -orig_dev_path $SPIDER_DIR/my/dev/human_test/human_test.json \


# USKG-regular
# -model_name hkunlp/from_all_T5_base_prefix_spider_with_cell_value2 \

# USKG-ASR (local)
# -model_name /Users/mac/Desktop/syt/Deep-Learning/Repos/UnifiedSKG/output/server_runs/A-T5_base_prefix_spider_with_cell_value-asr_mixed/checkpoint-66500 \

# USKG-ASR (server)
# -model_name /vault/uskg/output/A-T5_base_prefix_spider_with_cell_value-asr_mixed/checkpoint-66500 \


