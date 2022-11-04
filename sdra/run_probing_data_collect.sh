PROJ_DIR='/home/yshao/Projects'
DATASET='spider'

python -m sdra.probing_data_collect \
-ds ${DATASET} \
-dataset_dir ${PROJ_DIR}/SDR-analysis/data/spider \
-tables_path ${PROJ_DIR}/language/language/xsp/data/spider/tables.json \
-db_path ${PROJ_DIR}/language/language/xsp/data/spider/database \
-pb_in_dir ${PROJ_DIR}/SDR-analysis/data/probing/text2sql/link_prediction/${DATASET}/uskg \
-pb_out_dir ${PROJ_DIR}/SDR-analysis/data/probing/text2sql/link_prediction/${DATASET}/uskg-tmp

# -cfg Salesforce/T5_large_finetune_spider_with_cell_value.cfg
# -model t5-large-rd

