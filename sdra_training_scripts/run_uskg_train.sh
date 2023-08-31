python train.py \
--seed 2 \
--cfg Salesforce/A-T5_base_prefix_webqsp_sr.cfg \
--run_name A-T5_base_prefix_webqsp_sr \
--logging_strategy steps \
--logging_first_step true \
--logging_steps 4 \
--evaluation_strategy steps \
--eval_steps 500 \
--metric_for_best_model avr \
--greater_is_better true \
--save_strategy steps \
--save_steps 500 \
--save_total_limit 1 \
--load_best_model_at_end \
--gradient_accumulation_steps 4 \
--num_train_epochs 400 \
--adafactor true \
--learning_rate 5e-5 \
--do_train \
--do_eval \
--do_predict \
--predict_with_generate \
--output_dir output/A-T5_base_prefix_webqsp_sr-short \
--overwrite_output_dir \
--per_device_train_batch_size 8 \
--per_device_eval_batch_size 16 \
--generation_num_beams 1 \
--generation_max_length 128 \
--input_max_length 512 \
--ddp_find_unused_parameters true

## According to paper appendix: input_max_length=1024, batch_size=32, beam_size=4
# trying t5-large:
#   len=1024, can't run (even bsize=1 gets OOM)
#   len=512, max bsize=2
# trying t5-base:
#   len=1024, max bsize=2
#   len=512, max bsize=8
