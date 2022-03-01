set -e 

# SPIDER_DIR=/vault/spider
# PREDS_IN_DIR=~/SpeakQL/SpeakQL/Allennlp_models/outputs
# PREDS_OUT_DIR=/vault/SpeakQL/Allennlp_models/outputs/uskg-test-save

SPIDER_DIR=/Users/mac/Desktop/syt/Deep-Learning/Dataset/spider
PREDS_IN_DIR=/Users/mac/Desktop/syt/Deep-Learning/Projects-M/SpeakQL/SpeakQL/Allennlp_models/outputs
PREDS_OUT_DIR=/Users/mac/Desktop/syt/Deep-Learning/Projects-M/SpeakQL/SpeakQL/Allennlp_models/outputs/uskg-test-save

mkdir -p $PREDS_OUT_DIR

python uskg_infer.py \
	-cfg Salesforce/T5_base_prefix_spider_with_cell_value.cfg \
	-model_name hkunlp/from_all_T5_base_prefix_spider_with_cell_value2 \
	-test_dataset_path $SPIDER_DIR/my/dev/test_rewriter+phonemes.json \
	-orig_dev_path $SPIDER_DIR/dev.json \
	-db_path $SPIDER_DIR/database \
	-eval_vers 2.12.1.0t-2.27.0.0i 2.12.1.1t-2.27.0.1i 2.12.1.2t-2.27.0.2i 2.12.1.3t-2.27.0.3i 2.12.1.4t-2.27.0.4i \
	-eval_in_dir $PREDS_IN_DIR \
	-eval_out_dir $PREDS_OUT_DIR
