set -ex

PROJ_DIR='/home/yshao/Projects'
# DATASET='wikisql'
DATASET='spider'
PROBE_TASK='link_prediction'
# PROBE_TASK='single_node_reconstruction'


python -m sdra.probing_data_collect \
-ds ${DATASET} \
-probe_task ${PROBE_TASK} \
-orig_dataset_dir ${PROJ_DIR}/language/language/xsp/data/${DATASET} \
-graph_dataset_dir ${PROJ_DIR}/SDR-analysis/data/${DATASET} \
-pb_out_dir ${PROJ_DIR}/SDR-analysis/data/probing/text2sql/${PROBE_TASK}/${DATASET}/uskg-tmp \
-enc_bsz 1 \
-pb_in_dir ${PROJ_DIR}/SDR-analysis/data/probing/text2sql/${PROBE_TASK}/${DATASET}/uskg  --gpu
# -cfg Salesforce/T5_large_finetune_spider_with_cell_value.cfg \
# -model t5-large


# python -m sdra.probing_data_collect \
# -ds ${DATASET} \
# -probe_task ${PROBE_TASK} \
# -orig_dataset_dir ${PROJ_DIR}/language/language/xsp/data/${DATASET} \
# -graph_dataset_dir ${PROJ_DIR}/SDR-analysis/data/${DATASET} \
# -pb_in_dir ${PROJ_DIR}/SDR-analysis/data/probing/text2sql/${PROBE_TASK}/${DATASET}/uskg \
# -pb_out_dir ${PROJ_DIR}/SDR-analysis/data/probing/text2sql/${PROBE_TASK}/${DATASET}/t5-large-tmp \
# -enc_bsz 1 \
# -cfg Salesforce/T5_large_finetune_spider_with_cell_value.cfg \
# -model t5-large


# python -m sdra.probing_data_collect \
# -ds ${DATASET} \
# -probe_task ${PROBE_TASK} \
# -orig_dataset_dir ${PROJ_DIR}/language/language/xsp/data/${DATASET} \
# -graph_dataset_dir ${PROJ_DIR}/SDR-analysis/data/${DATASET} \
# -pb_in_dir ${PROJ_DIR}/SDR-analysis/data/probing/text2sql/${PROBE_TASK}/${DATASET}/uskg \
# -pb_out_dir ${PROJ_DIR}/SDR-analysis/data/probing/text2sql/${PROBE_TASK}/${DATASET}/t5-large-rd \
# -enc_bsz 1 \
# -cfg Salesforce/T5_large_finetune_spider_with_cell_value.cfg \
# -model t5-large-rd




# -pb_in_dir ${PROJ_DIR}/SDR-analysis/data/probing/text2sql/${PROBE_TASK}/${DATASET}/uskg \
# -cfg Salesforce/T5_large_finetune_spider_with_cell_value.cfg
# -model t5-large
# -max_label_occ 5
# --gpu
