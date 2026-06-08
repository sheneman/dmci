#!/bin/bash
############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# slurm_submit.sh: 2 methods x 5 seeds = 10 tasks
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

#SBATCH --job-name=exp_d
#SBATCH --partition=eight
#SBATCH --array=0-9
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=8:00:00
#SBATCH --output=experiments/exp_d/logs/exp_d_%A_%a.out
#SBATCH --error=experiments/exp_d/logs/exp_d_%A_%a.err

# 2 methods x 5 seeds = 10 tasks
# Array index 0-4: gp_direct seeds 0-4
# Array index 5-9: gp_dmci seeds 0-4

cd "$HOME/src/nncompile"
source .venv/bin/activate

mkdir -p experiments/exp_d/logs experiments/exp_d/results

SEED=$((SLURM_ARRAY_TASK_ID % 5))

if [ $SLURM_ARRAY_TASK_ID -lt 5 ]; then
    METHOD="gp_direct"
else
    METHOD="gp_dmci"
fi

echo "Array task $SLURM_ARRAY_TASK_ID: method=$METHOD seed=$SEED"

python3 -u -m experiments.exp_d.exp_d \
    --method "$METHOD" \
    --seed "$SEED" \
    --output-dir experiments/exp_d/results
