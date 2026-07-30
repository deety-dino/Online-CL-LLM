
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

deepspeed --num_gpus=1 src/run_llama_new.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path meta-llama/Llama-2-7b-chat-hf \
   --data_dir CL_Benchmark \
   --task_order mnli,cb,wic,copa,qqp,boolq,rte,imdb,yelp,amazon,sst2,dbpedia,agnews,multirc,yahoo \
   --task_config_dir configs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0_configs/mnli \
   --output_dir logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/1-mnli \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 2 \
   --gradient_accumulation_steps 4 \
   --learning_rate 5e-05 \
   --attn_lr 0.0 \
   --num_train_epochs 5 \
   
   --run_name test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0 \
   --distances_temperature 1.0 \
   --distances_way Attention \
   --max_source_length 1024 \
   --max_target_length 50 \
   --generation_max_length 50 \
   --add_task_name False \
   --add_dataset_name False \
   --overwrite_output_dir \
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

rm -rf logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/1-mnli/checkpoint*



deepspeed --num_gpus=1 src/run_llama_new.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path meta-llama/Llama-2-7b-chat-hf \
   --previous_lora_path logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/1-mnli/saved_weights \
   --previous_lora_distribution_path logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/1-mnli/saved_weights \
   --data_dir CL_Benchmark \
   --task_order mnli,cb,wic,copa,qqp,boolq,rte,imdb,yelp,amazon,sst2,dbpedia,agnews,multirc,yahoo \
   --gen_data_dir generated_data/lora_gen_15datasets_t5_xl \
   --task_config_dir configs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0_configs/cb \
   --output_dir logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/2-cb \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 2 \
   --gradient_accumulation_steps 4 \
   --learning_rate 5e-05 \
   --attn_lr 0.0 \
   --max_steps 100 \
   
   --run_name test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0 \
   --distances_temperature 1.0 \
   --distances_way Attention \
   --max_source_length 1024 \
   --max_target_length 50 \
   --generation_max_length 50 \
   --add_task_name False \
   --add_dataset_name False \
   --overwrite_output_dir \
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

rm -rf logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/2-cb/checkpoint*



deepspeed --num_gpus=1 src/run_llama_new.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path meta-llama/Llama-2-7b-chat-hf \
   --previous_lora_path logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/1-mnli/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/2-cb/saved_weights \
   --previous_lora_distribution_path logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/1-mnli/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/2-cb/saved_weights \
   --data_dir CL_Benchmark \
   --task_order mnli,cb,wic,copa,qqp,boolq,rte,imdb,yelp,amazon,sst2,dbpedia,agnews,multirc,yahoo \
   --gen_data_dir generated_data/lora_gen_long_llama \
   --task_config_dir configs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0_configs/wic \
   --output_dir logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/3-wic \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 2 \
   --gradient_accumulation_steps 4 \
   --learning_rate 5e-05 \
   --attn_lr 0.0 \
   --num_train_epochs 5 \
   
   --run_name test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0 \
   --distances_temperature 1.0 \
   --distances_way Attention \
   --max_source_length 1024 \
   --max_target_length 50 \
   --generation_max_length 50 \
   --add_task_name False \
   --add_dataset_name False \
   --overwrite_output_dir \
   --overwrite_cache \
   --lr_scheduler_type constant \
   --warmup_steps 0 \
   --logging_strategy steps \
   --logging_steps 10 \
   --metric_for_best_model eval_exact_match_for_wic \
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

rm -rf logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/3-wic/checkpoint*



deepspeed --num_gpus=1 src/run_llama_new.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path meta-llama/Llama-2-7b-chat-hf \
   --previous_lora_path logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/1-mnli/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/2-cb/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/3-wic/saved_weights \
   --previous_lora_distribution_path logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/1-mnli/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/2-cb/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/3-wic/saved_weights \
   --data_dir CL_Benchmark \
   --task_order mnli,cb,wic,copa,qqp,boolq,rte,imdb,yelp,amazon,sst2,dbpedia,agnews,multirc,yahoo \
   --gen_data_dir generated_data/lora_gen_15datasets_t5_xl \
   --task_config_dir configs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0_configs/copa \
   --output_dir logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/4-copa \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 2 \
   --gradient_accumulation_steps 4 \
   --learning_rate 5e-05 \
   --attn_lr 0.0 \
   --max_steps 200 \
   
   --run_name test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0 \
   --distances_temperature 1.0 \
   --distances_way Attention \
   --max_source_length 1024 \
   --max_target_length 50 \
   --generation_max_length 50 \
   --add_task_name False \
   --add_dataset_name False \
   --overwrite_output_dir \
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

rm -rf logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/4-copa/checkpoint*



deepspeed --num_gpus=1 src/run_llama_new.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path meta-llama/Llama-2-7b-chat-hf \
   --previous_lora_path logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/1-mnli/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/2-cb/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/3-wic/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/4-copa/saved_weights \
   --previous_lora_distribution_path logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/1-mnli/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/2-cb/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/3-wic/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/4-copa/saved_weights \
   --data_dir CL_Benchmark \
   --task_order mnli,cb,wic,copa,qqp,boolq,rte,imdb,yelp,amazon,sst2,dbpedia,agnews,multirc,yahoo \
   --gen_data_dir generated_data/lora_gen_long_llama \
   --task_config_dir configs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0_configs/qqp \
   --output_dir logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/5-qqp \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 2 \
   --gradient_accumulation_steps 4 \
   --learning_rate 5e-05 \
   --attn_lr 0.0 \
   --num_train_epochs 5 \
   
   --run_name test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0 \
   --distances_temperature 1.0 \
   --distances_way Attention \
   --max_source_length 1024 \
   --max_target_length 50 \
   --generation_max_length 50 \
   --add_task_name False \
   --add_dataset_name False \
   --overwrite_output_dir \
   --overwrite_cache \
   --lr_scheduler_type constant \
   --warmup_steps 0 \
   --logging_strategy steps \
   --logging_steps 10 \
   --metric_for_best_model eval_exact_match_for_qqp \
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

rm -rf logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/5-qqp/checkpoint*



deepspeed --num_gpus=1 src/run_llama_new.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path meta-llama/Llama-2-7b-chat-hf \
   --previous_lora_path logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/1-mnli/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/2-cb/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/3-wic/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/4-copa/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/5-qqp/saved_weights \
   --previous_lora_distribution_path logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/1-mnli/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/2-cb/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/3-wic/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/4-copa/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/5-qqp/saved_weights \
   --data_dir CL_Benchmark \
   --task_order mnli,cb,wic,copa,qqp,boolq,rte,imdb,yelp,amazon,sst2,dbpedia,agnews,multirc,yahoo \
   --gen_data_dir generated_data/lora_gen_15datasets_t5_xl \
   --task_config_dir configs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0_configs/boolq \
   --output_dir logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/6-boolq \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 2 \
   --gradient_accumulation_steps 4 \
   --learning_rate 5e-05 \
   --attn_lr 0.0 \
   --max_steps 500 \
   
   --run_name test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0 \
   --distances_temperature 1.0 \
   --distances_way Attention \
   --max_source_length 1024 \
   --max_target_length 50 \
   --generation_max_length 50 \
   --add_task_name False \
   --add_dataset_name False \
   --overwrite_output_dir \
   --overwrite_cache \
   --lr_scheduler_type constant \
   --warmup_steps 0 \
   --logging_strategy steps \
   --logging_steps 10 \
   --metric_for_best_model eval_exact_match_for_boolq \
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

rm -rf logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/6-boolq/checkpoint*



deepspeed --num_gpus=1 src/run_llama_new.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path meta-llama/Llama-2-7b-chat-hf \
   --previous_lora_path logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/1-mnli/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/2-cb/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/3-wic/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/4-copa/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/5-qqp/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/6-boolq/saved_weights \
   --previous_lora_distribution_path logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/1-mnli/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/2-cb/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/3-wic/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/4-copa/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/5-qqp/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/6-boolq/saved_weights \
   --data_dir CL_Benchmark \
   --task_order mnli,cb,wic,copa,qqp,boolq,rte,imdb,yelp,amazon,sst2,dbpedia,agnews,multirc,yahoo \
   --gen_data_dir generated_data/lora_gen_long_llama \
   --task_config_dir configs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0_configs/rte \
   --output_dir logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/7-rte \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 2 \
   --gradient_accumulation_steps 4 \
   --learning_rate 5e-05 \
   --attn_lr 0.0 \
   --num_train_epochs 5 \
   
   --run_name test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0 \
   --distances_temperature 1.0 \
   --distances_way Attention \
   --max_source_length 1024 \
   --max_target_length 50 \
   --generation_max_length 50 \
   --add_task_name False \
   --add_dataset_name False \
   --overwrite_output_dir \
   --overwrite_cache \
   --lr_scheduler_type constant \
   --warmup_steps 0 \
   --logging_strategy steps \
   --logging_steps 10 \
   --metric_for_best_model eval_exact_match_for_rte \
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

rm -rf logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/7-rte/checkpoint*



deepspeed --num_gpus=1 src/run_llama_new.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path meta-llama/Llama-2-7b-chat-hf \
   --previous_lora_path logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/1-mnli/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/2-cb/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/3-wic/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/4-copa/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/5-qqp/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/6-boolq/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/7-rte/saved_weights \
   --previous_lora_distribution_path logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/1-mnli/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/2-cb/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/3-wic/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/4-copa/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/5-qqp/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/6-boolq/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/7-rte/saved_weights \
   --data_dir CL_Benchmark \
   --task_order mnli,cb,wic,copa,qqp,boolq,rte,imdb,yelp,amazon,sst2,dbpedia,agnews,multirc,yahoo \
   --gen_data_dir generated_data/lora_gen_15datasets_t5_xl \
   --task_config_dir configs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0_configs/imdb \
   --output_dir logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/8-imdb \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 2 \
   --gradient_accumulation_steps 4 \
   --learning_rate 5e-05 \
   --attn_lr 0.0 \
   --max_steps 250 \
   
   --run_name test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0 \
   --distances_temperature 1.0 \
   --distances_way Attention \
   --max_source_length 1024 \
   --max_target_length 50 \
   --generation_max_length 50 \
   --add_task_name False \
   --add_dataset_name False \
   --overwrite_output_dir \
   --overwrite_cache \
   --lr_scheduler_type constant \
   --warmup_steps 0 \
   --logging_strategy steps \
   --logging_steps 10 \
   --metric_for_best_model eval_exact_match_for_imdb \
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

rm -rf logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/8-imdb/checkpoint*



deepspeed --num_gpus=1 src/run_llama_new.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path meta-llama/Llama-2-7b-chat-hf \
   --previous_lora_path logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/1-mnli/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/2-cb/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/3-wic/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/4-copa/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/5-qqp/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/6-boolq/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/7-rte/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/8-imdb/saved_weights \
   --previous_lora_distribution_path logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/1-mnli/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/2-cb/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/3-wic/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/4-copa/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/5-qqp/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/6-boolq/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/7-rte/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/8-imdb/saved_weights \
   --data_dir CL_Benchmark \
   --task_order mnli,cb,wic,copa,qqp,boolq,rte,imdb,yelp,amazon,sst2,dbpedia,agnews,multirc,yahoo \
   --gen_data_dir generated_data/lora_gen_long_llama \
   --task_config_dir configs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0_configs/yelp \
   --output_dir logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/9-yelp \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 2 \
   --gradient_accumulation_steps 4 \
   --learning_rate 5e-05 \
   --attn_lr 0.0 \
   --num_train_epochs 5 \
   
   --run_name test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0 \
   --distances_temperature 1.0 \
   --distances_way Attention \
   --max_source_length 1024 \
   --max_target_length 50 \
   --generation_max_length 50 \
   --add_task_name False \
   --add_dataset_name False \
   --overwrite_output_dir \
   --overwrite_cache \
   --lr_scheduler_type constant \
   --warmup_steps 0 \
   --logging_strategy steps \
   --logging_steps 10 \
   --metric_for_best_model eval_exact_match_for_yelp \
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

rm -rf logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/9-yelp/checkpoint*



deepspeed --num_gpus=1 src/run_llama_new.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path meta-llama/Llama-2-7b-chat-hf \
   --previous_lora_path logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/1-mnli/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/2-cb/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/3-wic/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/4-copa/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/5-qqp/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/6-boolq/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/7-rte/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/8-imdb/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/9-yelp/saved_weights \
   --previous_lora_distribution_path logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/1-mnli/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/2-cb/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/3-wic/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/4-copa/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/5-qqp/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/6-boolq/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/7-rte/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/8-imdb/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/9-yelp/saved_weights \
   --data_dir CL_Benchmark \
   --task_order mnli,cb,wic,copa,qqp,boolq,rte,imdb,yelp,amazon,sst2,dbpedia,agnews,multirc,yahoo \
   --gen_data_dir generated_data/lora_gen_long_llama \
   --task_config_dir configs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0_configs/amazon \
   --output_dir logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/10-amazon \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 2 \
   --gradient_accumulation_steps 4 \
   --learning_rate 5e-05 \
   --attn_lr 0.0 \
   --num_train_epochs 5 \
   
   --run_name test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0 \
   --distances_temperature 1.0 \
   --distances_way Attention \
   --max_source_length 1024 \
   --max_target_length 50 \
   --generation_max_length 50 \
   --add_task_name False \
   --add_dataset_name False \
   --overwrite_output_dir \
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

rm -rf logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/10-amazon/checkpoint*



deepspeed --num_gpus=1 src/run_llama_new.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path meta-llama/Llama-2-7b-chat-hf \
   --previous_lora_path logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/1-mnli/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/2-cb/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/3-wic/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/4-copa/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/5-qqp/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/6-boolq/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/7-rte/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/8-imdb/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/9-yelp/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/10-amazon/saved_weights \
   --previous_lora_distribution_path logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/1-mnli/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/2-cb/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/3-wic/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/4-copa/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/5-qqp/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/6-boolq/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/7-rte/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/8-imdb/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/9-yelp/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/10-amazon/saved_weights \
   --data_dir CL_Benchmark \
   --task_order mnli,cb,wic,copa,qqp,boolq,rte,imdb,yelp,amazon,sst2,dbpedia,agnews,multirc,yahoo \
   --gen_data_dir generated_data/lora_gen_long_llama \
   --task_config_dir configs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0_configs/sst2 \
   --output_dir logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/11-sst2 \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 2 \
   --gradient_accumulation_steps 4 \
   --learning_rate 5e-05 \
   --attn_lr 0.0 \
   --num_train_epochs 5 \
   
   --run_name test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0 \
   --distances_temperature 1.0 \
   --distances_way Attention \
   --max_source_length 1024 \
   --max_target_length 50 \
   --generation_max_length 50 \
   --add_task_name False \
   --add_dataset_name False \
   --overwrite_output_dir \
   --overwrite_cache \
   --lr_scheduler_type constant \
   --warmup_steps 0 \
   --logging_strategy steps \
   --logging_steps 10 \
   --metric_for_best_model eval_exact_match_for_sst2 \
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

rm -rf logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/11-sst2/checkpoint*



deepspeed --num_gpus=1 src/run_llama_new.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path meta-llama/Llama-2-7b-chat-hf \
   --previous_lora_path logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/1-mnli/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/2-cb/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/3-wic/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/4-copa/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/5-qqp/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/6-boolq/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/7-rte/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/8-imdb/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/9-yelp/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/10-amazon/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/11-sst2/saved_weights \
   --previous_lora_distribution_path logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/1-mnli/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/2-cb/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/3-wic/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/4-copa/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/5-qqp/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/6-boolq/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/7-rte/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/8-imdb/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/9-yelp/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/10-amazon/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/11-sst2/saved_weights \
   --data_dir CL_Benchmark \
   --task_order mnli,cb,wic,copa,qqp,boolq,rte,imdb,yelp,amazon,sst2,dbpedia,agnews,multirc,yahoo \
   --gen_data_dir generated_data/lora_gen_15datasets_t5_xl \
   --task_config_dir configs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0_configs/dbpedia \
   --output_dir logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/12-dbpedia \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 2 \
   --gradient_accumulation_steps 4 \
   --learning_rate 5e-05 \
   --attn_lr 0.0 \
   --max_steps 200 \
   
   --run_name test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0 \
   --distances_temperature 1.0 \
   --distances_way Attention \
   --max_source_length 1024 \
   --max_target_length 50 \
   --generation_max_length 50 \
   --add_task_name False \
   --add_dataset_name False \
   --overwrite_output_dir \
   --overwrite_cache \
   --lr_scheduler_type constant \
   --warmup_steps 0 \
   --logging_strategy steps \
   --logging_steps 10 \
   --metric_for_best_model eval_exact_match_for_dbpedia \
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

rm -rf logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/12-dbpedia/checkpoint*



deepspeed --num_gpus=1 src/run_llama_new.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path meta-llama/Llama-2-7b-chat-hf \
   --previous_lora_path logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/1-mnli/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/2-cb/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/3-wic/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/4-copa/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/5-qqp/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/6-boolq/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/7-rte/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/8-imdb/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/9-yelp/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/10-amazon/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/11-sst2/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/12-dbpedia/saved_weights \
   --previous_lora_distribution_path logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/1-mnli/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/2-cb/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/3-wic/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/4-copa/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/5-qqp/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/6-boolq/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/7-rte/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/8-imdb/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/9-yelp/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/10-amazon/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/11-sst2/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/12-dbpedia/saved_weights \
   --data_dir CL_Benchmark \
   --task_order mnli,cb,wic,copa,qqp,boolq,rte,imdb,yelp,amazon,sst2,dbpedia,agnews,multirc,yahoo \
   --gen_data_dir generated_data/lora_gen_long_llama \
   --task_config_dir configs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0_configs/agnews \
   --output_dir logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/13-agnews \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 2 \
   --gradient_accumulation_steps 4 \
   --learning_rate 5e-05 \
   --attn_lr 0.0 \
   --num_train_epochs 5 \
   
   --run_name test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0 \
   --distances_temperature 1.0 \
   --distances_way Attention \
   --max_source_length 1024 \
   --max_target_length 50 \
   --generation_max_length 50 \
   --add_task_name False \
   --add_dataset_name False \
   --overwrite_output_dir \
   --overwrite_cache \
   --lr_scheduler_type constant \
   --warmup_steps 0 \
   --logging_strategy steps \
   --logging_steps 10 \
   --metric_for_best_model eval_exact_match_for_agnews \
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

rm -rf logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/13-agnews/checkpoint*



deepspeed --num_gpus=1 src/run_llama_new.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path meta-llama/Llama-2-7b-chat-hf \
   --previous_lora_path logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/1-mnli/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/2-cb/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/3-wic/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/4-copa/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/5-qqp/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/6-boolq/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/7-rte/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/8-imdb/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/9-yelp/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/10-amazon/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/11-sst2/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/12-dbpedia/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/13-agnews/saved_weights \
   --previous_lora_distribution_path logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/1-mnli/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/2-cb/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/3-wic/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/4-copa/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/5-qqp/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/6-boolq/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/7-rte/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/8-imdb/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/9-yelp/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/10-amazon/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/11-sst2/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/12-dbpedia/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/13-agnews/saved_weights \
   --data_dir CL_Benchmark \
   --task_order mnli,cb,wic,copa,qqp,boolq,rte,imdb,yelp,amazon,sst2,dbpedia,agnews,multirc,yahoo \
   --gen_data_dir generated_data/lora_gen_15datasets_t5_xl \
   --task_config_dir configs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0_configs/multirc \
   --output_dir logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/14-multirc \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 2 \
   --gradient_accumulation_steps 4 \
   --learning_rate 5e-05 \
   --attn_lr 0.0 \
   --max_steps 500 \
   
   --run_name test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0 \
   --distances_temperature 1.0 \
   --distances_way Attention \
   --max_source_length 1024 \
   --max_target_length 50 \
   --generation_max_length 50 \
   --add_task_name False \
   --add_dataset_name False \
   --overwrite_output_dir \
   --overwrite_cache \
   --lr_scheduler_type constant \
   --warmup_steps 0 \
   --logging_strategy steps \
   --logging_steps 10 \
   --metric_for_best_model eval_exact_match_for_multirc \
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

rm -rf logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/14-multirc/checkpoint*



deepspeed --num_gpus=1 src/run_llama_new.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path meta-llama/Llama-2-7b-chat-hf \
   --previous_lora_path logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/1-mnli/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/2-cb/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/3-wic/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/4-copa/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/5-qqp/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/6-boolq/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/7-rte/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/8-imdb/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/9-yelp/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/10-amazon/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/11-sst2/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/12-dbpedia/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/13-agnews/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/14-multirc/saved_weights \
   --previous_lora_distribution_path logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/1-mnli/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/2-cb/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/3-wic/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/4-copa/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/5-qqp/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/6-boolq/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/7-rte/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/8-imdb/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/9-yelp/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/10-amazon/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/11-sst2/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/12-dbpedia/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/13-agnews/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/14-multirc/saved_weights \
   --data_dir CL_Benchmark \
   --task_order mnli,cb,wic,copa,qqp,boolq,rte,imdb,yelp,amazon,sst2,dbpedia,agnews,multirc,yahoo \
   --gen_data_dir generated_data/lora_gen_long_llama \
   --task_config_dir configs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0_configs/yahoo \
   --output_dir logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/15-yahoo \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 2 \
   --gradient_accumulation_steps 4 \
   --learning_rate 5e-05 \
   --attn_lr 0.0 \
   --num_train_epochs 5 \
   
   --run_name test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0 \
   --distances_temperature 1.0 \
   --distances_way Attention \
   --max_source_length 1024 \
   --max_target_length 50 \
   --generation_max_length 50 \
   --add_task_name False \
   --add_dataset_name False \
   --overwrite_output_dir \
   --overwrite_cache \
   --lr_scheduler_type constant \
   --warmup_steps 0 \
   --logging_strategy steps \
   --logging_steps 10 \
   --metric_for_best_model eval_exact_match_for_yahoo \
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

rm -rf logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/15-yahoo/checkpoint*



deepspeed --num_gpus=1 src/run_llama_new_eval.py \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path meta-llama/Llama-2-7b-chat-hf \
   --previous_lora_path logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/1-mnli/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/2-cb/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/3-wic/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/4-copa/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/5-qqp/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/6-boolq/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/7-rte/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/8-imdb/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/9-yelp/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/10-amazon/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/11-sst2/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/12-dbpedia/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/13-agnews/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/14-multirc/saved_weights \
   --previous_lora_distribution_path logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/1-mnli/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/2-cb/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/3-wic/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/4-copa/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/5-qqp/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/6-boolq/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/7-rte/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/8-imdb/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/9-yelp/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/10-amazon/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/11-sst2/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/12-dbpedia/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/13-agnews/saved_weights,logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/14-multirc/saved_weights \
   --data_dir CL_Benchmark \
   --task_order mnli,cb,wic,copa,qqp,boolq,rte,imdb,yelp,amazon,sst2,dbpedia,agnews,multirc,yahoo \
   --gen_data_dir generated_data/lora_gen_long_llama \
   --task_config_dir configs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0_configs/yahoo \
   --output_dir logs_and_outputs/test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0/outputs/15-yahoo \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 2 \
   --gradient_accumulation_steps 4 \
   --learning_rate 5e-05 \
   --attn_lr 0.0 \
   --num_train_epochs 5 \
   
   --run_name test_llama_7b_long_our_8_1_4_Attention_1.0_train_top_1_test_top_1_train_top_p_-1.0_test_top_p_-1.0 \
   --distances_temperature 1.0 \
   --distances_way Attention \
   --max_source_length 1024 \
   --max_target_length 50 \
   --generation_max_length 50 \
   --add_task_name False \
   --add_dataset_name False \
   --overwrite_output_dir \
   --overwrite_cache \
   --lr_scheduler_type constant \
   --warmup_steps 0 \
   --logging_strategy steps \
   --logging_steps 10 \
   --metric_for_best_model eval_exact_match_for_yahoo \
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
