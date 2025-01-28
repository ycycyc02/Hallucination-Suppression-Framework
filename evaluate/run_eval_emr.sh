cd ./evaluate_code
seed=2024
gpu=0

dataset_dir=path_of_emr

now_time=$(command date +%m-%d-%H-%M --date="+8 hour")
echo "now time ${now_time}"

# chatglm3
model="path_of_chatglm3"
model_dtype="fp16"
template="chatglm3"
split_max_len=8192

# BenTaso
model="path_of_bentaso"
model_dtype="fp16"
template="BenTaso"
split_max_len=8192

# Huatuo-7b
model="path_of_huatuo_7b"
model_dtype="bf16"
template="Huatuo"
split_max_len=4096

# AlpaCare
model="path_of_alpacare"
model_dtype="fp16"
template="AlpaCare"
split_max_len=4096

# 本文模型
model="path_of_model"
model_dtype="fp16"
template="chatglm3"
split_max_len=8192


# 打印model和dataset的组合
echo "Model: $model"
echo "dataset_dir: $dataset_dir"
CUDA_VISIBLE_DEVICES=${gpu} python eval_emr.py \
    --model_name_or_path ${model} \
    --template ${template} \
    --dataset_dir ${dataset_dir} \
    --n_shot 0 \
    --n_avg 1 \
    --seed ${seed} \
    --use_vllm \
    --split_max_len ${split_max_len}
