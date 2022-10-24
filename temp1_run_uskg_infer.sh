set -e 

SPIDER_DIR=/vault/spider
PREDS_IN_DIR=~/SpeakQL/SpeakQL/Allennlp_models/outputs
PREDS_OUT_BASE_DIR=~/SpeakQL/SpeakQL/Allennlp_models/outputs/uskg-test-save
PREDS_OUT_LARGE_DIR=~/SpeakQL/SpeakQL/Allennlp_models/outputs/uskg-large-test-save


# SPIDER_DIR=/Users/mac/Desktop/syt/Deep-Learning/Dataset/spider
# PREDS_IN_DIR=/Users/mac/Desktop/syt/Deep-Learning/Projects-M/SpeakQL/SpeakQL/Allennlp_models/outputs
# PREDS_OUT_DIR=/Users/mac/Desktop/syt/Deep-Learning/Projects-M/SpeakQL/SpeakQL/Allennlp_models/outputs/uskg-test-save

mkdir -p $PREDS_OUT_BASE_DIR
mkdir -p $PREDS_OUT_LARGE_DIR




python uskg_infer_joint.py \
        -cfg Salesforce/T5_base_prefix_spider_with_cell_value.cfg \
        -model_name hkunlp/from_all_T5_base_prefix_spider_with_cell_value2 \
        -db_path $SPIDER_DIR/database \
        -eval_vers 2.12.1.4t-2.35.1.4i humantest-yshao-2.12.1.4t-2.35.1.4i \
        -eval_in_dir $PREDS_IN_DIR \
        -eval_out_dir $PREDS_OUT_BASE_DIR


python uskg_infer_joint.py \
        -cfg Salesforce/T5_large_prefix_spider_with_cell_value.cfg \
        -model_name hkunlp/from_all_T5_large_prefix_spider_with_cell_value2 \
        -db_path $SPIDER_DIR/database \
        -eval_vers 2.12.1.4t-2.35.1.4i humantest-yshao-2.12.1.4t-2.35.1.4i \
        -eval_in_dir $PREDS_IN_DIR \
        -eval_out_dir $PREDS_OUT_LARGE_DIR







