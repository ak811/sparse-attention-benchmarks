#!/bin/bash
#SBATCH --job-name=latency
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=00:30:00

# Load environment reliably
module load anaconda3
eval "$(conda shell.bash hook)"
conda activate fibottention_env

cd /users/akhalegh/sparse-attention-benchmarks-main
export PYTHONPATH="$(pwd):$PYTHONPATH"

# This runs the benchmark file which automatically iterates through Dense, Simulated, and True Sparse.
python benchmarks/benchmark_latency.py