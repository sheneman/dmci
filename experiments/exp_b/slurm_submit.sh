#!/bin/bash
############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# slurm_submit.sh: Experiment B SLURM submission (hand-authored reference run, path A)
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

#SBATCH --job-name=exp_b
#SBATCH --partition=eight
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=experiments/exp_b/logs/exp_b_%j.out
#SBATCH --error=experiments/exp_b/logs/exp_b_%j.err

cd "$HOME/src/nncompile"
source .venv/bin/activate

mkdir -p experiments/exp_b/logs experiments/exp_b/results

python3 -m experiments.exp_b.run_all \
    --output-dir experiments/exp_b/results \
    --skip-existing
