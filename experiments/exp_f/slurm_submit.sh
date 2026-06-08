#!/bin/bash
############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# slurm_submit.sh: Experiment F SLURM submission (LLM-in-the-loop model discovery)
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

#SBATCH --job-name=exp_f
#SBATCH --partition=eight
#SBATCH --array=0-3
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=16:00:00
#SBATCH --output=experiments/exp_f/logs/exp_f_%A_%a.out
#SBATCH --error=experiments/exp_f/logs/exp_f_%A_%a.err

module load python/3.11.10
cd "$HOME/src/nncompile"
source .venv/bin/activate

mkdir -p experiments/exp_f/logs experiments/exp_f/results experiments/exp_f/llm_cache

TASK=$SLURM_ARRAY_TASK_ID

echo "=============================="
echo "Experiment F: LLM-in-the-Loop"
echo "Target: $TASK"
echo "Host: $(hostname)"
echo "Python: $(python3 --version)"
echo "Start: $(date)"
echo "=============================="

for SEED in 0 1 2; do
    echo ""
    echo "--- Target $TASK, Seed $SEED ---"
    python3 -u -m experiments.exp_f.exp_f \
        --target "$TASK" \
        --seed "$SEED" \
        --output-dir experiments/exp_f/results
done

echo ""
echo "End: $(date)"
