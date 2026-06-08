#!/bin/bash
############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# slurm_submit.sh: Experiment G SLURM submission (compositional modeling)
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

#SBATCH --job-name=exp_g
#SBATCH --partition=eight
#SBATCH --array=0-2
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=experiments/exp_g/logs/exp_g_%A_%a.out
#SBATCH --error=experiments/exp_g/logs/exp_g_%A_%a.err

module load python/3.11.10
cd "$HOME/src/nncompile"
source .venv/bin/activate

mkdir -p experiments/exp_g/logs experiments/exp_g/results

PROBLEM=$SLURM_ARRAY_TASK_ID

echo "=================================="
echo "Experiment G: Compositional Modeling"
echo "Problem: $PROBLEM"
echo "Host: $(hostname)"
echo "Python: $(python3 --version)"
echo "Start: $(date)"
echo "=================================="

for SEED in 0 1 2 3 4; do
    echo ""
    echo "--- Problem $PROBLEM, Seed $SEED ---"
    python3 -u -m experiments.exp_g.exp_g \
        --problem "$PROBLEM" \
        --seed "$SEED" \
        --output-dir experiments/exp_g/results
done

echo ""
echo "End: $(date)"
