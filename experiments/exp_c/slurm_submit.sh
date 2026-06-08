#!/bin/bash
############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# slurm_submit.sh: Experiment C SLURM submission (recursive scientific models)
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

#SBATCH --job-name=exp_c
#SBATCH --partition=eight
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=experiments/exp_c/logs/exp_c_%j.out
#SBATCH --error=experiments/exp_c/logs/exp_c_%j.err

cd "$HOME/src/nncompile"
source .venv/bin/activate

mkdir -p experiments/exp_c/logs experiments/exp_c/results

python3 -m experiments.exp_c.run_all \
    --output-dir experiments/exp_c/results \
    --skip-existing
