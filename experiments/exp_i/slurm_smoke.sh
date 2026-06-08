#!/bin/bash
############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# slurm_smoke.sh: High-d feasibility probe (d=96,126) on a COMPUTE node, using the uv .venv (NOT conda).
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

#SBATCH --job-name=exp_i_smoke
#SBATCH --partition=eight
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=experiments/exp_i/logs/smoke_%j.out
#SBATCH --error=experiments/exp_i/logs/smoke_%j.err

# High-d feasibility probe (d=96,126) on a COMPUTE node, using the uv .venv (NOT conda).
# Self-diagnoses the interpreter + imports so we can confirm the env before the array.
set +e
cd "$HOME/src/nncompile"
module load python/3.11.11 2>&1
source .venv/bin/activate
mkdir -p experiments/exp_i/logs
echo "host=$(hostname)"
echo "python=$(which python)  ($(python --version 2>&1))"
echo "exe=$(python -c 'import sys; print(sys.executable)' 2>&1)"
python -c "import torch,scipy,numpy; print('torch',torch.__version__,'scipy',scipy.__version__,'numpy',numpy.__version__)" 2>&1
python -c "import jax; print('jax',jax.__version__)" 2>&1
echo "start=$(date)"
python -u -m experiments.exp_i.smoke_scaling
echo "end=$(date)"
