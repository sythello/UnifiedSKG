PROJ_DIR='/home/yshao/Projects'
DATASET='wikisql'
PROBE_TASK='single_node_reconstruction'

# gdb -ex r --args \
python -m sdra.probing_data_collect \
-ds ${DATASET} \
-probe_task ${PROBE_TASK} \
-orig_dataset_dir ${PROJ_DIR}/language/language/xsp/data/${DATASET} \
-graph_dataset_dir ${PROJ_DIR}/SDR-analysis/data/${DATASET} \
-pb_out_dir ${PROJ_DIR}/SDR-analysis/data/probing/text2sql/${PROBE_TASK}/${DATASET}/uskg \
-enc_bsz 4

# -pb_in_dir ${PROJ_DIR}/SDR-analysis/data/probing/text2sql/${PROBE_TASK}/${DATASET}/uskg \
# -cfg Salesforce/T5_large_finetune_spider_with_cell_value.cfg
# -model t5-large
# -max_label_occ 5


