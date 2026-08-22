#!/bin/bash
#SBATCH --job-name=head_ablation
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=01:00:00

# Load environment reliably
module load anaconda3
eval "$(conda shell.bash hook)"
conda activate fibottention_env

cd /users/akhalegh/sparse-attention-benchmarks-main
export PYTHONPATH="$(pwd):$PYTHONPATH"

# Set paths to the specific training runs
DENSE_CHECKPOINT="runs/none_cifar10_20260822_013908/checkpoint-last.pth"
FIBO_CHECKPOINT="runs/fibottention_cifar10_20260822_013908/checkpoint-last.pth"

echo "==========================================================="
echo "1. Evaluating Dense MHSA Redundancy"
echo "==========================================================="
python benchmarks/eval_head_ablation.py \
  --dataset c10 \
  --model vit_base_patch16 \
  --nb_classes 10 \
  --batch_size 128 \
  --attn-cfg configs/attention/vit_none.yaml \
  --finetune "$DENSE_CHECKPOINT"

echo ""
echo "==========================================================="
echo "2. Evaluating Fibottention Redundancy"
echo "==========================================================="
python benchmarks/eval_head_ablation.py \
  --dataset c10 \
  --model vit_base_patch16 \
  --nb_classes 10 \
  --batch_size 128 \
  --attn-cfg configs/attention/vit_fibottention.yaml \
  --finetune "$FIBO_CHECKPOINT"