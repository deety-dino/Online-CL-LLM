
#!/bin/bash
#SBATCH -J cl                           
#SBATCH -o cl-%j.out                       
#SBATCH -p compute 
#SBATCH -N 1                           
#SBATCH -t 20:00:00   
#SBATCH --mem 128G 
#SBATCH --gres=gpu:a100-sxm4-80gb:1

fuser -k /dev/nvidia*

export CUDA_DEVICE_ORDER="PCI_BUS_ID"

port=$(shuf -i25000-30000 -n1)  

deepspeed --num_gpus=2 src/run_llama_new.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path meta-llama/Llama-2-7b-chat-hf \
   --data_dir CL_Benchmark \
   --task_order yelp,amazon,mnli,cb,copa \
   --task_config_dir configs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0_configs/yelp \
   --output_dir logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/1-yelp \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 8 \
   --gradient_accumulation_steps 4 \
   --learning_rate 2e-05 \
   --attn_lr 0.0 \
   --num_train_epochs 1 \
   --fp16 \
   --load_in_8bit \
   --deepspeed configs/ds_configs/stage3.config \
   --run_name test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0 \
   --distances_temperature 1.0 \
   --distances_way Attention \
   --max_source_length 512\
   --max_target_length 50 \
   --generation_max_length 50 \
   --add_task_name False \
   --add_dataset_name False \
   --overwrite_output_dir \
   --max_predict_samples 200 \
   --overwrite_cache \
   --lr_scheduler_type constant \
   --warmup_steps 0 \
   --logging_strategy steps \
   --logging_steps 10 \
   --metric_for_best_model eval_exact_match \
   --evaluation_strategy steps \
   --save_strategy steps \
   --save_total_limit 1 \
   --lora_r 4 \
   --lora_alpha 32 \
   --lora_dropout 0.0 \
   --load_best_model_at_end \
   --data_replay_freq -1 \
   --replay_after_n_epoch 0 \
   --kl_ratio 2 \
   --attn_temperature 1 \
   --train_key_weight_top 1 \
   --test_key_weight_top 1 \
   --train_key_weight_top_p -1.0 \
   --test_key_weight_top_p -1.0 \
   --successor N

rm -rf logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/1-yelp/checkpoint*



deepspeed --num_gpus=2 src/run_llama_new.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path meta-llama/Llama-2-7b-chat-hf \
   --previous_lora_path logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/1-yelp/saved_weights \
   --previous_lora_distribution_path logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/1-yelp/saved_weights \
   --data_dir CL_Benchmark \
   --task_order yelp,amazon,mnli,cb,copa \
   --gen_data_dir generated_data/lora_gen_long_llama \
   --task_config_dir configs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0_configs/amazon \
   --output_dir logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/2-amazon \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 8 \
   --gradient_accumulation_steps 4 \
   --learning_rate 2e-05 \
   --attn_lr 0.0 \
   --num_train_epochs 1 \
   --fp16 \
   --load_in_8bit \
   --deepspeed configs/ds_configs/stage3.config \
   --run_name test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0 \
   --distances_temperature 1.0 \
   --distances_way Attention \
   --max_source_length 512\
   --max_target_length 50 \
   --generation_max_length 50 \
   --add_task_name False \
   --add_dataset_name False \
   --overwrite_output_dir \
   --max_predict_samples 200 \
   --overwrite_cache \
   --lr_scheduler_type constant \
   --warmup_steps 0 \
   --logging_strategy steps \
   --logging_steps 10 \
   --metric_for_best_model eval_exact_match_for_amazon \
   --evaluation_strategy steps \
   --save_strategy steps \
   --save_total_limit 1 \
   --load_best_model_at_end \
   --lora_r 4 \
   --lora_alpha 32 \
   --lora_dropout 0.0 \
   --data_replay_freq -1 \
   --replay_after_n_epoch 0 \
   --kl_ratio 2 \
   --attn_temperature 1 \
   --train_key_weight_top 1 \
   --test_key_weight_top 1 \
   --train_key_weight_top_p -1.0 \
   --test_key_weight_top_p -1.0 \
   --successor N

rm -rf logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/2-amazon/checkpoint*



deepspeed --num_gpus=2 src/run_llama_new.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path meta-llama/Llama-2-7b-chat-hf \
   --previous_lora_path logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/1-yelp/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/2-amazon/saved_weights \
   --previous_lora_distribution_path logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/1-yelp/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/2-amazon/saved_weights \
   --data_dir CL_Benchmark \
   --task_order yelp,amazon,mnli,cb,copa \
   --gen_data_dir generated_data/lora_gen_long_llama \
   --task_config_dir configs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0_configs/mnli \
   --output_dir logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/3-mnli \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 8 \
   --gradient_accumulation_steps 4 \
   --learning_rate 2e-05 \
   --attn_lr 0.0 \
   --num_train_epochs 1 \
   --fp16 \
   --load_in_8bit \
   --deepspeed configs/ds_configs/stage3.config \
   --run_name test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0 \
   --distances_temperature 1.0 \
   --distances_way Attention \
   --max_source_length 512\
   --max_target_length 50 \
   --generation_max_length 50 \
   --add_task_name False \
   --add_dataset_name False \
   --overwrite_output_dir \
   --max_predict_samples 200 \
   --overwrite_cache \
   --lr_scheduler_type constant \
   --warmup_steps 0 \
   --logging_strategy steps \
   --logging_steps 10 \
   --metric_for_best_model eval_exact_match_for_mnli \
   --evaluation_strategy steps \
   --save_strategy steps \
   --save_total_limit 1 \
   --load_best_model_at_end \
   --lora_r 4 \
   --lora_alpha 32 \
   --lora_dropout 0.0 \
   --data_replay_freq -1 \
   --replay_after_n_epoch 0 \
   --kl_ratio 2 \
   --attn_temperature 1 \
   --train_key_weight_top 1 \
   --test_key_weight_top 1 \
   --train_key_weight_top_p -1.0 \
   --test_key_weight_top_p -1.0 \
   --successor N

rm -rf logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/3-mnli/checkpoint*



deepspeed --num_gpus=2 src/run_llama_new.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path meta-llama/Llama-2-7b-chat-hf \
   --previous_lora_path logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/1-yelp/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/2-amazon/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/3-mnli/saved_weights \
   --previous_lora_distribution_path logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/1-yelp/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/2-amazon/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/3-mnli/saved_weights \
   --data_dir CL_Benchmark \
   --task_order yelp,amazon,mnli,cb,copa \
   --gen_data_dir generated_data/lora_gen_15datasets_t5_xl \
   --task_config_dir configs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0_configs/cb \
   --output_dir logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/4-cb \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 8 \
   --gradient_accumulation_steps 4 \
   --learning_rate 2e-05 \
   --attn_lr 0.0 \
   --max_steps 100 \
   --fp16 \
   --load_in_8bit \
   --deepspeed configs/ds_configs/stage3.config \
   --run_name test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0 \
   --distances_temperature 1.0 \
   --distances_way Attention \
   --max_source_length 512\
   --max_target_length 50 \
   --generation_max_length 50 \
   --add_task_name False \
   --add_dataset_name False \
   --overwrite_output_dir \
   --max_predict_samples 200 \
   --overwrite_cache \
   --lr_scheduler_type constant \
   --warmup_steps 0 \
   --logging_strategy steps \
   --logging_steps 10 \
   --metric_for_best_model eval_exact_match_for_cb \
   --evaluation_strategy steps \
   --save_strategy steps \
   --save_total_limit 1 \
   --load_best_model_at_end \
   --lora_r 4 \
   --lora_alpha 32 \
   --lora_dropout 0.0 \
   --data_replay_freq -1 \
   --replay_after_n_epoch 0 \
   --kl_ratio 2 \
   --attn_temperature 1 \
   --train_key_weight_top 1 \
   --test_key_weight_top 1 \
   --train_key_weight_top_p -1.0 \
   --test_key_weight_top_p -1.0 \
   --successor N

rm -rf logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/4-cb/checkpoint*



deepspeed --num_gpus=2 src/run_llama_new.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path meta-llama/Llama-2-7b-chat-hf \
   --previous_lora_path logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/1-yelp/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/2-amazon/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/3-mnli/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/4-cb/saved_weights \
   --previous_lora_distribution_path logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/1-yelp/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/2-amazon/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/3-mnli/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/4-cb/saved_weights \
   --data_dir CL_Benchmark \
   --task_order yelp,amazon,mnli,cb,copa \
   --gen_data_dir generated_data/lora_gen_15datasets_t5_xl \
   --task_config_dir configs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0_configs/copa \
   --output_dir logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/5-copa \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 8 \
   --gradient_accumulation_steps 4 \
   --learning_rate 2e-05 \
   --attn_lr 0.0 \
   --max_steps 200 \
   --fp16 \
   --load_in_8bit \
   --deepspeed configs/ds_configs/stage3.config \
   --run_name test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0 \
   --distances_temperature 1.0 \
   --distances_way Attention \
   --max_source_length 512\
   --max_target_length 50 \
   --generation_max_length 50 \
   --add_task_name False \
   --add_dataset_name False \
   --overwrite_output_dir \
   --max_predict_samples 200 \
   --overwrite_cache \
   --lr_scheduler_type constant \
   --warmup_steps 0 \
   --logging_strategy steps \
   --logging_steps 10 \
   --metric_for_best_model eval_exact_match_for_copa \
   --evaluation_strategy steps \
   --save_strategy steps \
   --save_total_limit 1 \
   --load_best_model_at_end \
   --lora_r 4 \
   --lora_alpha 32 \
   --lora_dropout 0.0 \
   --data_replay_freq -1 \
   --replay_after_n_epoch 0 \
   --kl_ratio 2 \
   --attn_temperature 1 \
   --train_key_weight_top 1 \
   --test_key_weight_top 1 \
   --train_key_weight_top_p -1.0 \
   --test_key_weight_top_p -1.0 \
   --successor N

rm -rf logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/5-copa/checkpoint*



deepspeed --num_gpus=2 src/run_llama_new.py \
   --do_predict \
   --do_eval \
   --predict_with_generate \
   --model_name_or_path meta-llama/Llama-2-7b-chat-hf \
   --previous_lora_path logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/1-yelp/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/2-amazon/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/3-mnli/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/4-cb/saved_weights \
   --previous_lora_distribution_path logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/1-yelp/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/2-amazon/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/3-mnli/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/4-cb/saved_weights \
   --data_dir CL_Benchmark \
   --task_order yelp,amazon,mnli,cb,copa \
   --gen_data_dir generated_data/lora_gen_long_llama \
   --task_config_dir configs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0_configs/copa \
   --output_dir logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/5-copa \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 8 \
   --gradient_accumulation_steps 4 \
   --learning_rate 2e-05 \
   --attn_lr 0.0 \
   --num_train_epochs 1 \
   --fp16 \
   --load_in_8bit \
   --deepspeed configs/ds_configs/stage3.config \
   --run_name test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0 \
   --distances_temperature 1.0 \
   --distances_way Attention \
   --max_source_length 512\
   --max_target_length 50 \
   --generation_max_length 50 \
   --add_task_name False \
   --add_dataset_name False \
   --overwrite_output_dir \
   --max_predict_samples 200 \
   --overwrite_cache \
   --lr_scheduler_type constant \
   --warmup_steps 0 \
   --logging_strategy steps \
   --logging_steps 10 \
   --metric_for_best_model eval_exact_match_for_copa \
   --evaluation_strategy steps \
   --save_strategy steps \
   --save_total_limit 1 \
   --load_best_model_at_end \
   --lora_r 4 \
   --lora_alpha 32 \
   --lora_dropout 0.0 \
   --data_replay_freq -1 \
   --replay_after_n_epoch 0 \
   --kl_ratio 2 \
   --attn_temperature 1 \
   --train_key_weight_top 1 \
   --test_key_weight_top 1 \
   --train_key_weight_top_p -1.0 \
   --test_key_weight_top_p -1.0 \
   --successor N
