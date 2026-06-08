#!/bin/bash
############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# slurm_diffsoc.sh: Experiment H: DiffSoc-S batching-benchmark SLURM job
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

#SBATCH --job-name=diffsoc_bench
#SBATCH --partition=borowiec
#SBATCH --nodelist=n124
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=experiments/exp_h/results/bench_diffsoc_%j.out
#SBATCH --error=experiments/exp_h/results/bench_diffsoc_%j.err

module load python/3.11.10
source .venv/bin/activate

export PYTHONUNBUFFERED=1

mkdir -p experiments/exp_h/results

echo "=== GPU Info ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

echo "=== PyTorch ==="
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')"
echo ""

echo "=== DiffSoc-S Benchmark: CPU + GPU ==="
python -m experiments.exp_h.bench_diffsoc --device all

echo "=== Done ==="
