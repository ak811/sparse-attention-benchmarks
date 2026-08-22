#!/bin/bash
#SBATCH --job-name=overlap_c10
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=09:00:00

eval "$(conda shell.bash hook)"
conda activate fibottention_env

cd /users/akhalegh/sparse-attention-benchmarks-main
export PYTHONPATH="$(pwd):$PYTHONPATH"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUT_DIR="runs/overlap_c10_${TIMESTAMP}"
mkdir -p "$OUT_DIR"

torchrun --nproc_per_node=1 --master_port=$((10000 + $RANDOM % 1000)) \
  -m main_finetune --dataset c10 --model vit_base_patch16 \
  --epochs 100 --cls_token --nb_classes 10 --batch_size 64 \
  --output_dir "$OUT_DIR" --log_dir "$OUT_DIR" \
  --blr 1e-3 --layer_decay 0.75 --weight_decay 0.05 \
  --drop_path 0.2 --reprob 0.25 \
  --attn-cfg configs/attention/fibottention_high_overlap.yaml > "$OUT_DIR/slurm-$SLURM_JOB_ID.out" 2>&1