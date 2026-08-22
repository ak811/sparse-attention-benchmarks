#!/bin/bash
#SBATCH --job-name=rand_cifar10
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

# Navigate to repo root
cd /users/akhalegh/sparse-attention-benchmarks-main
export PYTHONPATH="$(pwd):$PYTHONPATH"

# --- TIMESTAMP DYNAMIC DIRECTORY SETUP ---
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUT_DIR="runs/random_cifar10_${TIMESTAMP}"
mkdir -p "$OUT_DIR"
# ----------------------------------------

# Since SLURM's #SBATCH --output line is static, we can move or duplicate 
# the current slurm output, or handle it via a cleaner re-queue trick. 
# However, the easiest drop-in way to capture it in $OUT_DIR is to 
# redirect the execution script's stdout/stderr, while letting SLURM 
# output a placeholder that we immediately clean up or leverage.

echo "Starting ..."
echo "Output directory: $OUT_DIR"

# Execute and pipe output to both the terminal/slurm and a log file inside OUT_DIR
torchrun \
  --nproc_per_node=1 \
  --master_port=$((10000 + $RANDOM % 1000)) \
  -m main_finetune \
  --dataset c10 \
  --model vit_base_patch16 \
  --epochs 100 \
  --cls_token \
  --nb_classes 10 \
  --batch_size 64 \
  --output_dir "$OUT_DIR" \
  --log_dir "$OUT_DIR" \
  --blr 1e-3 \
  --layer_decay 0.75 \
  --weight_decay 0.05 \
  --drop_path 0.2 \
  --reprob 0.25 \
  --attn-cfg configs/attention/vit_random.yaml \
  > "$OUT_DIR/slurm-$SLURM_JOB_ID.out" 2>&1