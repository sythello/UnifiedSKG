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
        -eval_vers humantest-yshao-2.12.1.0t-2.33.9.0i \
        -eval_in_dir $PREDS_IN_DIR \
        -eval_out_dir $PREDS_OUT_BASE_DIR


python uskg_infer_joint.py \
        -cfg Salesforce/T5_large_prefix_spider_with_cell_value.cfg \
        -model_name hkunlp/from_all_T5_large_prefix_spider_with_cell_value2 \
        -db_path $SPIDER_DIR/database \
        -eval_vers humantest-yshao-2.12.1.0t-2.33.9.0i \
        -eval_in_dir $PREDS_IN_DIR \
        -eval_out_dir $PREDS_OUT_LARGE_DIR


## Human test eval
python uskg_infer_joint.py \
        -cfg Salesforce/T5_base_prefix_spider_with_cell_value.cfg \
        -model_name hkunlp/from_all_T5_base_prefix_spider_with_cell_value2 \
        -db_path $SPIDER_DIR/database \
        -eval_vers humantest-yshao-1.15.2.0 humantest-yshao-1.15.2.1 humantest-yshao-1.15.2.2 humantest-yshao-1.15.2.3 \
        -eval_in_dir $PREDS_IN_DIR \
        -eval_out_dir $PREDS_OUT_BASE_DIR


python uskg_infer_joint.py \
        -cfg Salesforce/T5_large_prefix_spider_with_cell_value.cfg \
        -model_name hkunlp/from_all_T5_large_prefix_spider_with_cell_value2 \
        -db_path $SPIDER_DIR/database \
        -eval_vers humantest-yshao-1.15.2.0 humantest-yshao-1.15.2.1 humantest-yshao-1.15.2.2 humantest-yshao-1.15.2.3 \
        -eval_in_dir $PREDS_IN_DIR \
        -eval_out_dir $PREDS_OUT_LARGE_DIR



# ## Dev eval
# python uskg_infer_joint.py \
#         -cfg Salesforce/T5_base_prefix_spider_with_cell_value.cfg \
#         -model_name hkunlp/from_all_T5_base_prefix_spider_with_cell_value2 \
#         -db_path $SPIDER_DIR/database \
#         -eval_vers 1.15.2.0 1.15.2.1 1.15.2.2 1.15.2.3 \
#         -eval_in_dir $PREDS_IN_DIR \
#         -eval_out_dir $PREDS_OUT_BASE_DIR \
#         --eval_in_prefix "dev-rewriter-" \
#         --dataset_out_prefix "dev-" \
#         --result_out_prefix "eval-dev-"


# python uskg_infer_joint.py \
#         -cfg Salesforce/T5_large_prefix_spider_with_cell_value.cfg \
#         -model_name hkunlp/from_all_T5_large_prefix_spider_with_cell_value2 \
#         -db_path $SPIDER_DIR/database \
#         -eval_vers 1.15.2.0 1.15.2.1 1.15.2.2 1.15.2.3 \
#         -eval_in_dir $PREDS_IN_DIR \
#         -eval_out_dir $PREDS_OUT_LARGE_DIR \
#         --eval_in_prefix "dev-rewriter-" \
#         --dataset_out_prefix "dev-" \
#         --result_out_prefix "eval-dev-"


# python uskg_infer_joint.py \
#         -cfg Salesforce/T5_large_prefix_spider_with_cell_value.cfg \
#         -model_name hkunlp/from_all_T5_large_prefix_spider_with_cell_value2 \
#         -db_path $SPIDER_DIR/database \
#         -eval_vers 2.12.1.1t-2.33.5.1i-freeze=ADJ \
#             2.12.1.1t-2.33.5.1i-freeze=ADP \
#             2.12.1.1t-2.33.5.1i-freeze=ADV \
#             2.12.1.1t-2.33.5.1i-freeze=AUX \
#             2.12.1.1t-2.33.5.1i-freeze=CCONJ \
#             2.12.1.1t-2.33.5.1i-freeze=DET \
#             2.12.1.1t-2.33.5.1i-freeze=NOUN \
#             2.12.1.1t-2.33.5.1i-freeze=NUM \
#             2.12.1.1t-2.33.5.1i-freeze=PART \
#             2.12.1.1t-2.33.5.1i-freeze=PRON \
#             2.12.1.1t-2.33.5.1i-freeze=PROPN \
#             2.12.1.1t-2.33.5.1i-freeze=PUNCT \
#             2.12.1.1t-2.33.5.1i-freeze=SCONJ \
#             2.12.1.1t-2.33.5.1i-freeze=VERB \
#         -eval_in_dir $PREDS_IN_DIR \
#         -eval_out_dir $PREDS_OUT_LARGE_DIR

# regular TTS
# -test_dataset_path $SPIDER_DIR/my/dev/test_rewriter+phonemes.json \
# -orig_dev_path $SPIDER_DIR/dev.json \

# human test
# -test_dataset_path $SPIDER_DIR/my/dev/human_test/human_test_yshao_rewriter.json \
# -orig_dev_path $SPIDER_DIR/my/dev/human_test/human_test.json \


# USKG-regular
# -model_name hkunlp/from_all_T5_base_prefix_spider_with_cell_value2 \

# USKG-ASR (local)
# -model_name /Users/mac/Desktop/syt/Deep-Learning/Repos/UnifiedSKG/output/server_runs/A-T5_base_prefix_spider_with_cell_value-asr_mixed/checkpoint-79500 \

# USKG-ASR (server)
# -model_name /vault/uskg/output/A-T5_base_prefix_spider_with_cell_value-asr_mixed/checkpoint-79500 \




