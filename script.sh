#!/bin/bash
source ~/.bashrc
source activate fibottention_env

id=$1
out_dir=$2
model=$3
dataset=$4
classes=$5
device=$6
batch=$7
attn_name=${8:-vit_sparse_local}   # e.g. vit_sparse_local, vit_fibottention, ...

attn_cfg_path="image_classification/configs/attention/${attn_name}.yaml"

# cd to repo root (this assumes script.sh sits at repo root; adjust if needed)
cd "$(dirname "$0")"

# Make sure package is importable
export PYTHONPATH="$(pwd):$PYTHONPATH"

mkdir -p "$out_dir"

{
  echo "Experiment ID: $id"
  echo "Dataset: $dataset"
  echo "Model: vit_${model}_patch16"
  echo "Classes: $classes"
  echo "Device: $device"
  echo "Batch Size: $batch"
  echo "Attention CFG: $attn_cfg_path"
  echo "----------------------------------"
} >> "$out_dir/log.txt"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
  --nproc_per_node=8 --master_port=$((10000 + id)) \
  -m main_finetune \
  --dataset "$dataset" --model "vit_${model}_patch16" \
  --epochs 100 \
  --cls_token \
  --nb_classes "$classes" \
  --batch_size "$batch" \
  --output_dir "$out_dir" \
  --log_dir "$out_dir" \
  --blr 1e-3 --layer_decay 0.75 \
  --weight_decay 0.05 --drop_path 0.2 --reprob 0.25 \
  --attn-cfg "$attn_cfg_path"
