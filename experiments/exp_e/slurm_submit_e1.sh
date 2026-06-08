#!/bin/bash
############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# slurm_submit_e1.sh: 12 tasks, one per array index (each runs 20 restarts internally)
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

#SBATCH --job-name=exp_e1
#SBATCH --partition=eight
#SBATCH --array=0-11
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=experiments/exp_e/logs/exp_e1_%A_%a.out
#SBATCH --error=experiments/exp_e/logs/exp_e1_%A_%a.err

# 12 tasks, one per array index (each runs 20 restarts internally)

cd "$HOME/src/nncompile"
source .venv/bin/activate

mkdir -p experiments/exp_e/logs experiments/exp_e/results

TASK=$SLURM_ARRAY_TASK_ID

echo "Array task $SLURM_ARRAY_TASK_ID: task=$TASK"
echo "Host: $(hostname)"
echo "Start: $(date)"

python3 -u -m experiments.exp_e.exp_e1 \
    --task "$TASK" \
    --output-dir experiments/exp_e/results

echo "End: $(date)"
