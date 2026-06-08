#!/bin/bash
############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# slurm_part_e.sh: Experiment H: Part-E benchmark SLURM job
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

#SBATCH --job-name=exp_h_compile
#SBATCH --partition=borowiec
#SBATCH --nodelist=n124
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=experiments/exp_h/results/exp_h_part_e_%j.out
#SBATCH --error=experiments/exp_h/results/exp_h_part_e_%j.err

module load python/3.11.10
source .venv/bin/activate

mkdir -p experiments/exp_h/results

echo "=== GPU Info ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

echo "=== PyTorch compile backend ==="
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}, compile available: {hasattr(torch, \"compile\")}')"
echo ""

echo "=== Part E: torch.compile Speedup (GPU) ==="
python -m experiments.exp_h.exp_h --part E --device cuda

echo "=== Part E: torch.compile Speedup (CPU) ==="
python -m experiments.exp_h.exp_h --part E --device cpu

echo "=== Done ==="
