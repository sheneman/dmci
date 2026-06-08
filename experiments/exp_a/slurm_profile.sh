#!/bin/bash
############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# slurm_profile.sh: Profile the 10x DMCI overhead decomposition (CPU only)
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

#SBATCH --job-name=profile_10x
#SBATCH --partition=eight
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=experiments/exp_a/logs/profile_%j.out
#SBATCH --error=experiments/exp_a/logs/profile_%j.err

# Profile the 10x DMCI overhead decomposition (CPU only)

cd "$HOME/src/nncompile"
source .venv/bin/activate

mkdir -p experiments/exp_a/results experiments/exp_a/logs

python3 -u -m experiments.exp_a.profile_decomposition \
    --n-iters 100 \
    2>&1 | tee experiments/exp_a/results/profile_decomposition.txt
