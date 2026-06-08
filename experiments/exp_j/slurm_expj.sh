#!/bin/bash
############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# slurm_expj.sh: Exp J: Program-Space Calibration — DMCI vs. compile-each-program (JAX/lambdify).
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

#SBATCH --job-name=exp_j
#SBATCH --partition=eight
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=experiments/exp_j/logs/expj_%j.out
#SBATCH --error=experiments/exp_j/logs/expj_%j.err

# Exp J: Program-Space Calibration — DMCI vs. compile-each-program (JAX/lambdify).
# Four curves vs N (number of distinct programs) at recursive fractions {0%, 100%}:
# cumulative compile, per-structure engineering, coverage, matched recovery error.
# CPU partition; 32G because the JAX arms hold N jitted executables at high N (the
# memory/scalability point the experiment measures).

module load python/3.11.10
cd "$HOME/src/nncompile"
source .venv/bin/activate
mkdir -p experiments/exp_j/logs experiments/exp_j/results

echo "Exp J  host=$(hostname)  python=$(python3 --version)  start=$(date)"
python3 -u -m experiments.exp_j.run_expj \
    --Ns 1 100 10000 --fractions 0 1 \
    --recover-sample 30 --recover-budget 30 \
    --output-dir experiments/exp_j/results
echo "end=$(date)"
