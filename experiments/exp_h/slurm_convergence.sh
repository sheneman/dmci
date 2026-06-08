#!/bin/bash
############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# slurm_convergence.sh: Experiment H: convergence-benchmark SLURM job
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

#SBATCH --job-name=diffesm_converge
#SBATCH --partition=borowiec
#SBATCH --nodelist=n124
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=experiments/exp_h/results/convergence_%j.out
#SBATCH --error=experiments/exp_h/results/convergence_%j.err

module load python/3.11.10
source .venv/bin/activate

mkdir -p experiments/exp_h/results

echo "=== Node Info ==="
hostname
echo ""

echo "=== PyTorch ==="
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')"
echo ""

echo "=== Convergence Benchmark: CPU ==="
python -m experiments.exp_h.bench_convergence --device cpu

echo "=== Done ==="
