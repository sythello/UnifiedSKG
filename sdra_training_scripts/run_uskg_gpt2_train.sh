RUN_NAME=A-GPT2_small_prefix_spider_with_cell_value

mkdir -p output/$RUN_NAME

python train.py \
--seed 2 \
--cfg Salesforce/A-GPT2_small_prefix_spider_with_cell_value.cfg \
--run_name $RUN_NAME \
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
--num_train_epochs 400 \
--adafactor true \
--learning_rate 5e-5 \
--do_train \
--do_eval \
--do_predict \
--predict_with_generate \
--output_dir output/$RUN_NAME \
--overwrite_output_dir \
--per_device_train_batch_size 4 \
--per_device_eval_batch_size 16 \
--generation_num_beams 1 \
--generation_max_length 128 \
--input_max_length 502 \
--ddp_find_unused_parameters true
# | tee output/$RUN_NAME/training-log.txt

## According to paper appendix: input_max_length=512, batch_size=32, beam_size=1
## Now using input_max_length 502 = 512 - 10 (prefix_len)

## Update:
## Now using input_max_length = 400, generation_max_length = 100
## Now input_max_length only stands for "input", not concatenated sequence
