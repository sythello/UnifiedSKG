RUN_NAME=A-GPT2_small_prefix_spider_with_cell_value-pfx=20
RUN_ID=20231107

mkdir -p output/$RUN_NAME

python train.py \
--seed 2 \
--cfg Salesforce/A-GPT2_small_prefix_spider_with_cell_value-pfx=20.cfg \
--run_name $RUN_NAME \
--overwrite_output_dir \
--logging_strategy steps \
--logging_first_step true \
--logging_steps 4 \
--evaluation_strategy steps \
--eval_steps 500 \
--metric_for_best_model avr \
--save_strategy steps \
--save_steps 500 \
--save_total_limit 2 \
--load_best_model_at_end \
--greater_is_better true \
--gradient_accumulation_steps 8 \
--num_train_epochs 400 \
--adafactor true \
--learning_rate 5e-5 \
--do_train \
--do_eval \
--do_predict \
--predict_with_generate \
--output_dir output/$RUN_NAME/run-$RUN_ID \
--per_device_train_batch_size 4 \
--per_device_eval_batch_size 16 \
--generation_num_beams 1 \
--generation_max_length 490 \
--input_max_length 362 \
--ddp_find_unused_parameters true \
--is_causal_lm true


## Back-up Args
# 



# Can't use `tee output/$RUN_NAME/training-log.txt`: tee interferes with input()

## According to paper appendix: input_max_length=512, batch_size=32, beam_size=1
## Notice: generation_max_length need to be the length of "generated seq"; for gpt2, it's input + output
## Now using input_max_length = 372, generation_max_length = 500
## Now input_max_length only stands for "input", not concatenated sequence

# About greater_is_better: if using loss, lower is better. Provided is using 'avr' which is SQL eval
# Set save_total_limit 2 to save the last ckpt, just in case

# Added --is_causal_lm true to also provide info for trainer post-processing
# (It is also specified in model_args)
