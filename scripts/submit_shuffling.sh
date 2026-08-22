#!/bin/bash
#SBATCH --job-name=shuffle_c10
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=09:00:00

# Load environment reliably
module load anaconda3
eval "$(conda shell.bash hook)"
conda activate fibottention_env

cd /users/akhalegh/sparse-attention-benchmarks-main
export PYTHONPATH="$(pwd):$PYTHONPATH"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUT_DIR_TRUE="runs/shuffle_TRUE_c10_${TIMESTAMP}"
mkdir -p "$OUT_DIR_TRUE"

# Tests the new Shuffled configuration (your old vit_fibottention.yaml had shuffled: false)
torchrun --nproc_per_node=1 --master_port=$((10000 + $RANDOM % 1000)) \
  -m main_finetune --dataset c10 --model vit_base_patch16 \
  --epochs 100 --cls_token --nb_classes 10 --batch_size 64 \
  --output_dir "$OUT_DIR_TRUE" --log_dir "$OUT_DIR_TRUE" \
  --blr 1e-3 --layer_decay 0.75 --weight_decay 0.05 \
  --drop_path 0.2 --reprob 0.25 \
  --attn-cfg configs/attention/vit_fibottention_shuffled.yaml > "$OUT_DIR_TRUE/slurm-true.out" 2>&1