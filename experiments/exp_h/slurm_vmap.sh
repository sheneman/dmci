#!/bin/bash
############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# slurm_vmap.sh: Exp H vmap baseline: DMCI batched throughput vs jax.vmap of the directly-compiled model.
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

#SBATCH --job-name=exp_h_vmap
#SBATCH --partition=eight
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=experiments/exp_h/logs/vmap_%j.out
#SBATCH --error=experiments/exp_h/logs/vmap_%j.err

# Exp H vmap baseline: DMCI batched throughput vs jax.vmap of the directly-compiled model.
# Single CPU core parity with the paper's CPU narrative; jax[cpu] installed in the venv.

module load python/3.11.10
cd "$HOME/src/nncompile"
source .venv/bin/activate
mkdir -p experiments/exp_h/logs experiments/exp_h/results
export JAX_PLATFORMS=cpu

echo "Exp H vmap  host=$(hostname)  python=$(python3 --version)  start=$(date)"
python3 -u -m experiments.exp_h.bench_vmap \
    --pop-sizes 1 10 100 \
    --output experiments/exp_h/results/vmap_results.json
echo "end=$(date)"
