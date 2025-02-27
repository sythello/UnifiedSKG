python train.py \
--seed 2 \
--cfg Salesforce/A-T5_base_prefix_spider_with_cell_value-asr_mixed.cfg \
--run_name A-T5_base_prefix_spider_with_cell_value-asr_mixed \
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
--gradient_accumulation_steps 8 \
--num_train_epochs 60 \
--adafactor true \
--learning_rate 5e-5 \
--do_train \
--do_eval \
--do_predict \
--predict_with_generate \
--output_dir output/A-T5_base_prefix_spider_with_cell_value-asr_mixed \
--overwrite_output_dir \
--per_device_train_batch_size 4 \
--per_device_eval_batch_size 16 \
--generation_num_beams 1 \
--generation_max_length 128 \
--input_max_length 512 \
--ddp_find_unused_parameters true

## According to paper appendix: input_max_length=512, batch_size=32, beam_size=1
## Changed num_train_epochs 400 -> 60 to keep to total steps about the same (since training set size changes: 7000 -> 48114)
## Changed num_train_epochs 60 -> 30 to make the total training time similar to ILM (~2 days)
## ^ incorrect; changed num_train_epochs back to 60, since ILM takes about 2.5 days and tagger takes about 0.5 days; more importantly, num_train_epochs is related to lr scheduling, and changing to 30 makes it overall worse

