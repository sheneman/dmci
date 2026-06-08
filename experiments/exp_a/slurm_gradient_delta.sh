#!/bin/bash
############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# slurm_gradient_delta.sh: Experiment A: SLURM job for the gradient-delta sub-experiment (autograd vs finite-difference)
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

#SBATCH --job-name=grad_delta
#SBATCH --partition=eight
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=2:00:00
#SBATCH --output=experiments/exp_a/logs/gradient_delta_%j.out
#SBATCH --error=experiments/exp_a/logs/gradient_delta_%j.err

cd "$HOME/src/nncompile"
source .venv/bin/activate

mkdir -p experiments/exp_a/logs experiments/exp_a/results

python3 -u -m experiments.exp_a.gradient_delta 2>&1 | tee experiments/exp_a/results/gradient_delta.txt
