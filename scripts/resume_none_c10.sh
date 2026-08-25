#!/bin/bash
#SBATCH --job-name=res_none_c10
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=09:00:00
#SBATCH --array=0-1

# Load environment reliably
module load anaconda3
eval "$(conda shell.bash hook)"
conda activate fibottention_env
cd /users/akhalegh/sparse-attention-benchmarks-main
export PYTHONPATH="$(pwd):$PYTHONPATH"

# Define the runs to resume
RUNS=(
    "runs/none_cifar10_20260824_223222"
    "runs/none_cifar10_20260824_223242"
)

# Grab the specific directory for THIS array task (0 or 1)
OUT_DIR="${RUNS[$SLURM_ARRAY_TASK_ID]}"
CHECKPOINT="$OUT_DIR/checkpoint-last.pth"

# Redirect output to a new SLURM log file inside the existing directory
exec > "$OUT_DIR/slurm-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out" 2>&1

# Delete the empty default log SLURM created in the repo root
rm -f "slurm-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out"

echo "======================================================================="
echo "Resuming Dense (None) Attention Training on CIFAR-10..."
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Resuming from: $CHECKPOINT"
echo "======================================================================="

torchrun --nproc_per_node=1 --master_port=$((10000 + $RANDOM % 1000)) \
  -m main_finetune --dataset c10 --model vit_base_patch16 --epochs 100 \
  --cls_token --nb_classes 10 --batch_size 64 \
  --output_dir "$OUT_DIR" --log_dir "$OUT_DIR" \
  --blr 1e-3 --layer_decay 0.75 --weight_decay 0.05 \
  --drop_path 0.2 --reprob 0.25 \
  --attn-cfg configs/attention/vit_none.yaml \
  --resume "$CHECKPOINT"