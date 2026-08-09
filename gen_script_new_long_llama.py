import json
import os
import pathlib
import numpy as np
from copy import deepcopy

# --- HÀM BẢO VỆ CHỐNG LỖI CHIA CHO 0 ---
def safe_temp(value, eps=1e-8):
    """Tránh giá trị temperature bằng 0 gây lỗi ZeroDivisionError/NaN khi chia logits."""
    return max(float(value), eps)

def safe_div(numerator, denominator, default=0.0):
    """Hàm chia an toàn cho tính toán metrics/loss."""
    return numerator / denominator if denominator != 0 else default

# --- HÀM ĐỌC/GHI JSON & JSONL ---
def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def write_json(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_jsonline(path):
    result = []
    if not os.path.exists(path):
        return result
    with open(path, 'r', encoding='utf-8') as f:
        for line_s in f:
            line_s = line_s.strip()
            if line_s:
                result.append(json.loads(line_s))
    return result

def write_jsonline(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        for line in data:
            f.write(json.dumps(line, ensure_ascii=False) + '\n')

# --- CẤU HÌNH TÁC VỤ ---
order_idx = 3

if order_idx == 4:
    all_tasks = [
        "yelp", "amazon", "mnli", "cb", "copa", "qqp", "rte",
        "imdb", "sst2", "dbpedia", "agnews", "yahoo", "multirc", "boolq", "wic"
    ]
else:
    all_tasks = [
        "mnli", "cb", "wic", "copa", "qqp", "boolq", "rte",
        "imdb", "yelp", "amazon", "sst2", "dbpedia", "agnews", "multirc", "yahoo"
    ]

dataset_list = all_tasks
if not dataset_list:
    raise ValueError("Danh sách dataset_list không được để rỗng!")

task_order = ','.join(all_tasks)
config_template = {"Long_Sequence": []}

# --- HYPERPARAMETERS (ĐÃ BỌC SAFE TEMP) ---
lora_r = 4
lora_alpha = 32
lora_dropout = 0.0
kl_ratio = 2
learning_rate = 5e-5
num_train_epochs = 3
attn_lr = 0.0
replay_after_n_epoch = 0

# Kiểm tra an toàn cho các tham số chia logits / softmax
attn_temperature = safe_temp(1)
distances_temperature = safe_temp(1.0)

train_top = 1
test_top = train_top
train_top_p = -1.0
test_top_p = -1.0
successor = 'N'

run_name = f"test_llama_7b_long_our_8_1_4_{distances_way_str if 'distances_way_str' in locals() else 'Attention'}_{distances_temperature}_train_top_{train_top}_test_top_{test_top}_train_top_p_{train_top_p}_test_top_p_{test_top_p}"
model_path = 'meta-llama/Llama-2-7b-chat-hf'

distances_way = 'Attention'

# --- TẠO CONFIG FILES ---
history_config = []
for one_data_name in dataset_list:
    pathlib.Path(f'./configs/{run_name}_configs/{one_data_name}').mkdir(parents=True, exist_ok=True)

    config = {
        "sampling strategy": "full",
        "dataset name": f"{one_data_name}"
    } 
    history_config.append(config)

    dev_config = deepcopy(config_template)
    dev_config['Long_Sequence'].append(config)
    write_json(f'./configs/{run_name}_configs/{one_data_name}/dev_tasks.json', dev_config)
    
    train_config = deepcopy(config_template)
    train_config['Long_Sequence'].append(config)
    write_json(f'./configs/{run_name}_configs/{one_data_name}/train_tasks.json', train_config)

    test_config = deepcopy(config_template)
    test_config['Long_Sequence'].extend(history_config)
    write_json(f'./configs/{run_name}_configs/{one_data_name}/test_tasks.json', test_config)

# --- CHUẨN BỊ BASH SCRIPT (DÙNG TORCHRUN THAY CHO DEEPSPEED) ---
sh_str = rf'''#!/bin/bash
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

torchrun --nproc_per_node=2 src/run_llama_new.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path {model_path} \
   --data_dir CL_Benchmark \
   --task_order {task_order} \
   --task_config_dir configs/{run_name}_configs/{dataset_list[0]} \
   --output_dir logs_and_outputs/{run_name}/outputs/1-{dataset_list[0]} \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 8 \
   --gradient_accumulation_steps 4 \
   --learning_rate {learning_rate} \
   --attn_lr {attn_lr} \
   --num_train_epochs {num_train_epochs} \
   --fp16 \
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
   --successor {successor}

rm -rf logs_and_outputs/{run_name}/outputs/1-{dataset_list[0]}/checkpoint*
'''

previous_lora_path_list = []
for idx in range(len(dataset_list) - 1):
    previous_lora_path_list.append(f"logs_and_outputs/{run_name}/outputs/{idx+1}-{dataset_list[idx]}/saved_weights")
    previous_lora_path = ','.join(previous_lora_path_list)
    next_task = dataset_list[idx + 1]
    
    if next_task in ["cb", "copa", "boolq", "imdb", "dbpedia", "multirc"]:
        max_steps_dict = {"cb": 100, "copa": 200, "boolq": 500, "imdb": 250, "dbpedia": 200}
        max_steps = max_steps_dict.get(next_task, 500)
        
        sh_str += rf'''
torchrun --nproc_per_node=2 src/run_llama_new.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path {model_path} \
   --previous_lora_path {previous_lora_path} \
   --previous_lora_distribution_path {previous_lora_path} \
   --data_dir CL_Benchmark \
   --task_order {task_order} \
   --gen_data_dir generated_data/lora_gen_15datasets_t5_xl \
   --task_config_dir configs/{run_name}_configs/{next_task} \
   --output_dir logs_and_outputs/{run_name}/outputs/{idx+2}-{next_task} \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 8 \
   --gradient_accumulation_steps 4 \
   --learning_rate {learning_rate} \
   --attn_lr {attn_lr} \
   --max_steps {max_steps} \
   --fp16 \
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
   --metric_for_best_model eval_exact_match_for_{next_task} \
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
   --successor {successor}

rm -rf logs_and_outputs/{run_name}/outputs/{idx+2}-{next_task}/checkpoint*
'''
    else:
        sh_str += rf'''
torchrun --nproc_per_node=2 src/run_llama_new.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path {model_path} \
   --previous_lora_path {previous_lora_path} \
   --previous_lora_distribution_path {previous_lora_path} \
   --data_dir CL_Benchmark \
   --task_order {task_order} \
   --gen_data_dir generated_data/lora_gen_long_llama \
   --task_config_dir configs/{run_name}_configs/{next_task} \
   --output_dir logs_and_outputs/{run_name}/outputs/{idx+2}-{next_task} \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 8 \
   --gradient_accumulation_steps 4 \
   --learning_rate {learning_rate} \
   --attn_lr {attn_lr} \
   --num_train_epochs {num_train_epochs} \
   --fp16 \
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
   --metric_for_best_model eval_exact_match_for_{next_task} \
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
   --successor {successor}

rm -rf logs_and_outputs/{run_name}/outputs/{idx+2}-{next_task}/checkpoint*
'''

with open(f'{run_name}.sh', 'w', encoding='utf-8') as f:
    f.write(sh_str)