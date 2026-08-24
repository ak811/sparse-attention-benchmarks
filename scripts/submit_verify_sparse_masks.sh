#!/bin/bash
#SBATCH --job-name=verify_fibo_masks
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00

# Load environment reliably
module load anaconda3
eval "$(conda shell.bash hook)"
conda activate fibottention_env

# Navigate to repo root (Update if needed)
cd /users/akhalegh/sparse-attention-benchmarks-main
export PYTHONPATH="$(pwd):$PYTHONPATH"

# Execute the python script from the benchmarks directory
python benchmarks/verify_sparse_masks.py