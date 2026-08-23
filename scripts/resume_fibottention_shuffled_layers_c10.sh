#!/bin/bash
#SBATCH --job-name=res_fibo_l_c10
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=09:00:00

module load anaconda3
eval "$(conda shell.bash hook)"
conda activate fibottention_env
cd /users/akhalegh/sparse-attention-benchmarks-main
export PYTHONPATH="$(pwd):$PYTHONPATH"

# Point directly to the existing output directory
OUT_DIR="runs/fibo_layers_c10_20260822_204637"
CHECKPOINT="$OUT_DIR/checkpoint-last.pth"

# Redirect output to a new SLURM log file inside the existing directory
exec > "$OUT_DIR/slurm-$SLURM_JOB_ID.out" 2>&1
rm -f "slurm-$SLURM_JOB_ID.out"

echo "Resuming Fibottention (Shuffled Layers) Training on CIFAR-10..."
echo "Resuming from: $CHECKPOINT"

torchrun --nproc_per_node=1 --master_port=$((10000 + $RANDOM % 1000)) \
  -m main_finetune --dataset c10 --model vit_base_patch16 --epochs 100 \
  --cls_token --nb_classes 10 --batch_size 64 \
  --output_dir "$OUT_DIR" --log_dir "$OUT_DIR" \
  --blr 1e-3 --layer_decay 0.75 --weight_decay 0.05 \
  --drop_path 0.2 --reprob 0.25 \
  --attn-cfg configs/attention/vit_fibottention_shuffled_layers.yaml \
  --resume "$CHECKPOINT"