#!/bin/bash
############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# slurm_submit.sh: Part C first (fast, validates correctness)
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

#SBATCH --job-name=exp_h_batch
#SBATCH --partition=borowiec
#SBATCH --nodelist=n124
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=experiments/exp_h/results/exp_h_%j.out
#SBATCH --error=experiments/exp_h/results/exp_h_%j.err

module load python/3.11.10
source .venv/bin/activate

mkdir -p experiments/exp_h/results

echo "=== GPU Info ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

echo "=== Running Experiment H: All Parts ==="

# Part C first (fast, validates correctness)
echo "--- Part C: Correctness ---"
python -m experiments.exp_h.exp_h --part C --device cuda

# Part A: Forward throughput
echo "--- Part A: Throughput ---"
python -m experiments.exp_h.exp_h --part A --device cuda

# Also run CPU baseline for Part A
echo "--- Part A: CPU baseline ---"
python -m experiments.exp_h.exp_h --part A --device cpu

# Part B: Training speedup
echo "--- Part B: Training ---"
python -m experiments.exp_h.exp_h --part B --device cuda

# Part D: Population batching
echo "--- Part D: Population ---"
python -m experiments.exp_h.exp_h --part D --device cuda

echo "=== Done ==="
