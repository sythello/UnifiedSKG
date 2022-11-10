PROJ_DIR='/home/yshao/Projects'
DATASET='wikisql'

python -m sdra.probing_data_collect \
-ds ${DATASET} \
-orig_dataset_dir ${PROJ_DIR}/language/language/xsp/data/${DATASET} \
-graph_dataset_dir ${PROJ_DIR}/SDR-analysis/data/${DATASET} \
-pb_in_dir ${PROJ_DIR}/SDR-analysis/data/probing/text2sql/link_prediction/${DATASET}/uskg \
-pb_out_dir ${PROJ_DIR}/SDR-analysis/data/probing/text2sql/link_prediction/${DATASET}/t5-large-rd \
-cfg Salesforce/T5_large_finetune_spider_with_cell_value.cfg \
-model t5-large-rd

# -pb_in_dir ${PROJ_DIR}/SDR-analysis/data/probing/text2sql/link_prediction/${DATASET}/uskg
# -cfg Salesforce/T5_large_finetune_spider_with_cell_value.cfg
# -model t5-large

