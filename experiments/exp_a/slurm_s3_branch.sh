#!/bin/bash
############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# slurm_s3_branch.sh: S3: Branch-dependent constant experiment
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

#SBATCH --job-name=exp_s3_branch
#SBATCH --partition=eight
#SBATCH --array=0-2
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=4:00:00
#SBATCH --output=experiments/exp_a/logs/s3_branch_%A_%a.out
#SBATCH --error=experiments/exp_a/logs/s3_branch_%A_%a.err

# S3: Branch-dependent constant experiment
# Task 0: direct compilation training (10 seeds)
# Task 1: DMCI training (10 seeds)
# Task 2: basin-of-attraction study (21 alpha sweep points)

cd "$HOME/src/nncompile"
source .venv/bin/activate

mkdir -p experiments/exp_a/logs experiments/exp_a/results

case $SLURM_ARRAY_TASK_ID in
    0)
        echo "Task 0: S3 branch — direct training"
        python3 -u -m experiments.exp_a.exp_s3_branch \
            --mode train --method direct \
            --output-dir experiments/exp_a/results
        ;;
    1)
        echo "Task 1: S3 branch — DMCI training"
        python3 -u -m experiments.exp_a.exp_s3_branch \
            --mode train --method dmci \
            --output-dir experiments/exp_a/results
        ;;
    2)
        echo "Task 2: S3 branch — basin of attraction"
        python3 -u -m experiments.exp_a.exp_s3_branch \
            --mode basin \
            --output-dir experiments/exp_a/results
        ;;
esac

echo "Task $SLURM_ARRAY_TASK_ID complete."
