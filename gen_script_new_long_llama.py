import json
def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def write_json(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False,indent=2)

def load_jsonline(path):
    with open(path, 'r', encoding='utf-8') as f:
        result=[]
        for line_s in f:
            line=json.loads(line_s)
            result.append(line)
    return result

def write_jsonline(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        for line in data:
            line_s=json.dumps(line, ensure_ascii=False)
            f.write(line_s)
            f.write('\n')

order_idx = 3

if order_idx == 4:
    all_tasks=[
        "yelp",
        "amazon",
        "mnli",
        "cb",
        "copa",
        "qqp",
        "rte",
        "imdb",
        "sst2",
        "dbpedia",
        "agnews",
        "yahoo",
        "multirc",
        "boolq",
        "wic"
    ] # Order 4
else:
    all_tasks = ["mnli",
                 "cb",
                 "wic",
                 "copa",
                 "qqp",
                 "boolq",
                 "rte",
                 "imdb",
                 "yelp",
                 "amazon",
                 "sst2",
                 "dbpedia",
                 "agnews",
                 "multirc",
                 "yahoo"] # Order 3

dataset_list = all_tasks
task_order = ','.join(all_tasks)

config_template={
    "Long_Sequence": [
    ],
}

import os
import pathlib
import numpy as np
from copy import deepcopy

# Kaggle-friendly defaults: reduce VRAM/RAM usage and training cost.
lora_r = 4
lora_alpha = 16
lora_dropout = 0.05
kl_ratio = 2
attn_temperature = 1
learning_rate = 5e-5
num_train_epochs = 3
attn_lr = 0.
replay_after_n_epoch = 0

# Lower memory footprint for 2x15G GPUs / 30GiB RAM.
max_source_length = 256
max_target_length = 32
generation_max_length = 32
per_device_train_batch_size = 1
per_device_eval_batch_size = 2
gradient_accumulation_steps = 8
gradient_checkpointing = True


distances_temperature=1.0
distances_way='Attention'
train_top=1
test_top=train_top
train_top_p=-1.0
test_top_p=-1.0

successor='N'

run_name = f"test_llama_7b_long_our_8_1_4_{distances_way}_{distances_temperature}_train_top_{train_top}_test_top_{test_top}_train_top_p_{train_top_p}_test_top_p_{test_top_p}"
model_path='meta-llama/Llama-2-7b-chat-hf'

history_config=[]
for one_data_name in dataset_list:

    pathlib.Path(f'./configs/{run_name}_configs/{one_data_name}').mkdir(parents=True, exist_ok=True)

    config={
        "sampling strategy": "full",
        "dataset name": f"{one_data_name}"
    } 
    history_config.append(config)

    dev_config=deepcopy(config_template)
    dev_config['Long_Sequence'].append(config)
    write_json(f'./configs/{run_name}_configs/{one_data_name}/dev_tasks.json', dev_config)
    
    train_config=deepcopy(config_template)
    train_config['Long_Sequence'].append(config)
    write_json(f'./configs/{run_name}_configs/{one_data_name}/train_tasks.json', train_config)

    test_config=deepcopy(config_template)
    test_config['Long_Sequence'].extend(history_config)
    write_json(f'./configs/{run_name}_configs/{one_data_name}/test_tasks.json', test_config)

sh_str=rf'''
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
    --model_name_or_path {model_path} \
    --load_in_4bit True \
   --data_dir CL_Benchmark \
   --task_order {task_order} \
   --task_config_dir configs/{run_name}_configs/{dataset_list[0]} \
   --output_dir logs_and_outputs/{run_name}/outputs/1-{dataset_list[0]} \
   --per_device_train_batch_size {per_device_train_batch_size} \
   --per_device_eval_batch_size {per_device_eval_batch_size} \
   --gradient_accumulation_steps {gradient_accumulation_steps} \
   --max_num_instances_per_task 2000 \
   --max_num_instances_per_eval_task 100 \
   --learning_rate {learning_rate} \
   --attn_lr {attn_lr} \
   --num_train_epochs {num_train_epochs} \
   
   --deepspeed configs/ds_configs/stage2.config \
   --run_name {run_name} \
   --distances_temperature {distances_temperature} \
   --distances_way {distances_way} \
   --max_source_length {max_source_length} \
   --max_target_length {max_target_length} \
   --generation_max_length {generation_max_length} \
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
   --lora_r {lora_r} \
   --lora_alpha {lora_alpha} \
   --lora_dropout {lora_dropout} \
   --load_best_model_at_end \
   --data_replay_freq -1 \
   --replay_after_n_epoch 0 \
   --kl_ratio {kl_ratio} \
   --attn_temperature {attn_temperature} \
   --train_key_weight_top {train_top} \
   --test_key_weight_top {test_top} \
   --train_key_weight_top_p {train_top_p} \
   --test_key_weight_top_p {test_top_p} \
   --successor {successor} \
   --low_cpu_mem_usage True

rm -rf logs_and_outputs/{run_name}/outputs/1-{dataset_list[0]}/checkpoint*

'''
# ==========================================
# 1. BẮT BUỘC: Reset danh sách đường dẫn về RỖNG trước tiên
# ==========================================
previous_lora_path_list = []
previous_lora_path = ""

# ==========================================
# 2. TẠO CÂU LỆNH CHO TASK 1 (mnli)
# ==========================================
# Đảm bảo trong câu lệnh sh_str của mnli KHÔNG CÓ tham số --previous_lora_path
sh_str += rf'''
python3.10 -u -m deepspeed.launcher.launch --master_port 29500 src/run_llama_new.py \
    --do_train \
    --do_predict \
    --model_name_or_path meta-llama/Llama-2-7b-chat-hf \
    --load_in_4bit True \
    --data_dir CL_Benchmark \
    --task_config_dir configs/long_bench/mnli \
    --output_dir logs_and_outputs/{run_name}/outputs/1-mnli \
    --deepspeed configs/ds_configs/stage2.config \
    --max_source_length 256 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --max_steps 200 \
    --learning_rate 5e-4 \
    --lora_r 4
'''
# (Lưu ý: Tuyệt đối KHÔNG cho --previous_lora_path vào đoạn template mnli ở đây)


# ==========================================
# 3. ĐOẠN VÒNG LẶP CỦA BẠN (Dùng cho từ Task 2 - cb trở đi)
# ==========================================
for idx in range(len(dataset_list)-1):

    # Thêm weights của task vừa hoàn thành vào danh sách
    previous_lora_path_list.append(f"logs_and_outputs/{run_name}/outputs/{idx+1}-{dataset_list[idx]}/saved_weights")
    previous_lora_path = ','.join(previous_lora_path_list)
    
    if dataset_list[idx+1] in ["cb", "copa", "boolq", "imdb", "dbpedia", "multirc"]:
        if dataset_list[idx+1] == "cb":
            max_steps = 80
        elif dataset_list[idx+1] == "copa":
            max_steps = 120
        elif dataset_list[idx+1] == "boolq":
            max_steps = 200
        elif dataset_list[idx+1] == "imdb":
            max_steps = 120
        elif dataset_list[idx+1] == "dbpedia":
            max_steps = 100
        else:
            max_steps = 200
        
        # Trong template này MỚI DÙNG --previous_lora_path '{previous_lora_path}'
        sh_str += rf'''

deepspeed --num_gpus=1 src/run_llama_new.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
    --model_name_or_path {model_path} \
    --load_in_4bit True \
    --previous_lora_path {previous_lora_path} \
   --previous_lora_distribution_path {previous_lora_path} \
   --data_dir CL_Benchmark \
   --task_order {task_order} \
   --gen_data_dir generated_data/lora_gen_15datasets_t5_xl \
   --task_config_dir configs/{run_name}_configs/{dataset_list[idx+1]} \
   --output_dir logs_and_outputs/{run_name}/outputs/{idx+2}-{dataset_list[idx+1]} \
   --per_device_train_batch_size {per_device_train_batch_size} \
   --per_device_eval_batch_size {per_device_eval_batch_size} \
   --gradient_accumulation_steps {gradient_accumulation_steps} \
   --max_num_instances_per_task 2000 \
   --max_num_instances_per_eval_task 100 \
   --learning_rate {learning_rate} \
   --attn_lr {attn_lr} \
   --max_steps {max_steps} \
   
   --deepspeed configs/ds_configs/stage2.config \
   --run_name {run_name} \
   --distances_temperature {distances_temperature} \
   --distances_way {distances_way} \
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
   --metric_for_best_model eval_exact_match_for_{dataset_list[idx+1]} \
   --evaluation_strategy steps \
   --save_strategy steps \
   --save_total_limit 1 \
   --load_best_model_at_end \
   --lora_r {lora_r} \
   --lora_alpha {lora_alpha} \
   --lora_dropout {lora_dropout} \
   --data_replay_freq -1 \
   --replay_after_n_epoch {replay_after_n_epoch} \
   --kl_ratio {kl_ratio} \
   --attn_temperature {attn_temperature} \
   --train_key_weight_top {train_top} \
   --test_key_weight_top {test_top} \
   --train_key_weight_top_p {train_top_p} \
   --test_key_weight_top_p {test_top_p} \
   --successor {successor} \
   --low_cpu_mem_usage True

rm -rf logs_and_outputs/{run_name}/outputs/{idx+2}-{dataset_list[idx+1]}/checkpoint*

'''
    else:
        sh_str+=rf'''

deepspeed --num_gpus=1 src/run_llama_new.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
    --model_name_or_path {model_path} \
    --load_in_4bit True \
    --previous_lora_path {previous_lora_path} \
   --previous_lora_distribution_path {previous_lora_path} \
   --data_dir CL_Benchmark \
   --task_order {task_order} \
   --gen_data_dir generated_data/lora_gen_long_llama \
   --task_config_dir configs/{run_name}_configs/{dataset_list[idx+1]} \
   --output_dir logs_and_outputs/{run_name}/outputs/{idx+2}-{dataset_list[idx+1]} \
   --per_device_train_batch_size {per_device_train_batch_size} \
   --per_device_eval_batch_size {per_device_eval_batch_size} \
   --gradient_accumulation_steps {gradient_accumulation_steps} \
   --max_num_instances_per_task 2000 \
   --max_num_instances_per_eval_task 100 \
   --learning_rate {learning_rate} \
   --attn_lr {attn_lr} \
   --num_train_epochs {num_train_epochs} \
   
   --deepspeed configs/ds_configs/stage2.config \
   --run_name {run_name} \
   --distances_temperature {distances_temperature} \
   --distances_way {distances_way} \
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
   --metric_for_best_model eval_exact_match_for_{dataset_list[idx+1]} \
   --evaluation_strategy steps \
   --save_strategy steps \
   --save_total_limit 1 \
   --load_best_model_at_end \
   --lora_r {lora_r} \
   --lora_alpha {lora_alpha} \
   --lora_dropout {lora_dropout} \
   --data_replay_freq -1 \
   --replay_after_n_epoch {replay_after_n_epoch} \
   --kl_ratio {kl_ratio} \
   --attn_temperature {attn_temperature} \
   --train_key_weight_top {train_top} \
   --test_key_weight_top {test_top} \
   --train_key_weight_top_p {train_top_p} \
   --test_key_weight_top_p {test_top_p} \
   --successor {successor} \
   --low_cpu_mem_usage True

rm -rf logs_and_outputs/{run_name}/outputs/{idx+2}-{dataset_list[idx+1]}/checkpoint*

'''


sh_str+=rf'''

deepspeed --num_gpus=1 src/run_llama_new.py \
    --do_predict \
    --predict_with_generate \
    --model_name_or_path {model_path} \
    --load_in_4bit True \
    --previous_lora_path {previous_lora_path} \
    --previous_lora_distribution_path {previous_lora_path} \
   --data_dir CL_Benchmark \
   --task_order {task_order} \
   --gen_data_dir generated_data/lora_gen_long_llama \
   --task_config_dir configs/{run_name}_configs/{dataset_list[idx+1]} \
   --output_dir logs_and_outputs/{run_name}/outputs/{idx+2}-{dataset_list[idx+1]} \
   --per_device_train_batch_size {per_device_train_batch_size} \
   --per_device_eval_batch_size {per_device_eval_batch_size} \
   --gradient_accumulation_steps {gradient_accumulation_steps} \
   --max_num_instances_per_task 2000 \
   --max_num_instances_per_eval_task 100 \
   --learning_rate {learning_rate} \
   --attn_lr {attn_lr} \
   --num_train_epochs {num_train_epochs} \
   
   --deepspeed configs/ds_configs/stage2.config \
   --run_name {run_name} \
   --distances_temperature {distances_temperature} \
   --distances_way {distances_way} \
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
   --metric_for_best_model eval_exact_match_for_{dataset_list[idx+1]} \
   --evaluation_strategy steps \
   --save_strategy steps \
   --save_total_limit 1 \
   --load_best_model_at_end \
   --lora_r {lora_r} \
   --lora_alpha {lora_alpha} \
   --lora_dropout {lora_dropout} \
   --data_replay_freq -1 \
   --replay_after_n_epoch {replay_after_n_epoch} \
   --kl_ratio {kl_ratio} \
   --attn_temperature {attn_temperature} \
   --train_key_weight_top {train_top} \
   --test_key_weight_top {test_top} \
   --train_key_weight_top_p {train_top_p} \
   --test_key_weight_top_p {test_top_p} \
   --successor {successor} \
   --low_cpu_mem_usage True
'''
    
with open(f'{run_name}.sh', 'w') as f:
    f.write(sh_str)