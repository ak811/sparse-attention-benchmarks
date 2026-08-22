#!/bin/bash
#SBATCH --job-name=head_ablation
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=01:00:00

eval "$(conda shell.bash hook)"
conda activate fibottention_env

cd /users/akhalegh/sparse-attention-benchmarks-main
export PYTHONPATH="$(pwd):$PYTHONPATH"

# TODO: Adjust to point to the actual trained checkpoint from your old Fibottention runs
FIBO_CHECKPOINT="exp/checkpoint-best.pth"

# 1. Evaluate redundancy of Fibottention
echo "Evaluating Fibottention Redundancy..."
python eval_head_ablation.py \
  --dataset c10 \
  --model vit_base_patch16 \
  --nb_classes 10 \
  --batch_size 128 \
  --attn-cfg configs/attention/vit_fibottention.yaml \
  --finetune "$FIBO_CHECKPOINT"

# Note: You should also run this script pointing to a Dense (vit_none.yaml) checkpoint to get the comparison for the table.