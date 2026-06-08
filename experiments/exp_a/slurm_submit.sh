#!/bin/bash
############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# slurm_submit.sh: Run full experiment serially (graph cache shared across seeds)
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

#SBATCH --job-name=exp_a
#SBATCH --partition=sheneman
#SBATCH --nodelist=n128
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=experiments/exp_a/logs/exp_a_%j.out
#SBATCH --error=experiments/exp_a/logs/exp_a_%j.err

# Run full experiment serially (graph cache shared across seeds)
# 5 methods x 6 programs x 10 seeds = 300 runs

cd "$HOME/src/nncompile"
source .venv/bin/activate

mkdir -p experiments/exp_a/logs experiments/exp_a/results

python3 -m experiments.exp_a.run_all \
    --output-dir experiments/exp_a/results \
    --skip-existing
