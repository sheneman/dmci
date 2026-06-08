#!/bin/bash
############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# slurm_array.sh: Experiment C SLURM array (recursive scientific models)
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
#SBATCH --array=0-7
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=experiments/exp_c/logs/exp_c_%A_%a.out
#SBATCH --error=experiments/exp_c/logs/exp_c_%A_%a.err

MODELS=(
    C01_lotka_volterra
    C02_sir_epidemic
    C03_decay_chain
    C04_logistic_map
    C05_continued_fraction
    C06_damped_pendulum
    C07_iir_filter
    C08_cascaded_ema
)

MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}

cd "$HOME/src/nncompile"
source .venv/bin/activate

mkdir -p experiments/exp_c/logs experiments/exp_c/results

echo "Starting model $MODEL (array task $SLURM_ARRAY_TASK_ID)"

python3 -u -m experiments.exp_c.run_all \
    --model "$MODEL" \
    --output-dir experiments/exp_c/results \
    --skip-existing
