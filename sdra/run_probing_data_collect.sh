PROJ_DIR='/home/yshao/Projects'

python -m sdra.probing_data_collect \
-in_spider_dir ${PROJ_DIR}/SDR-analysis/data/spider \
-in_tables ${PROJ_DIR}/language/language/xsp/data/spider/tables.json \
-in_dbs_dir ${PROJ_DIR}/language/language/xsp/data/spider/database \
-pb_in_dir ${PROJ_DIR}/SDR-analysis/data/probing/text2sql/link_prediction/spider/uskg \
-pb_out_dir ${PROJ_DIR}/SDR-analysis/data/probing/text2sql/link_prediction/spider/t5-large-rd \
-cfg Salesforce/T5_large_finetune_spider_with_cell_value.cfg \
-model t5-large-rd

