#!/bin/bash
############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# slurm_comparison.sh: Exp I (re-scoped): exact-gradient (reparam + L-BFGS + multi-start) vs differential
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

#SBATCH --job-name=exp_i_cmp
#SBATCH --partition=eight
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=experiments/exp_i/logs/comparison_%j.out
#SBATCH --error=experiments/exp_i/logs/comparison_%j.err

# Exp I (re-scoped): exact-gradient (reparam + L-BFGS + multi-start) vs differential
# evolution at equal wall-clock, swept over parameter count. CPU partition (scalar
# interpreter; the paper shows CPU > GPU here). Args: [method=dmci|direct] [budget_s].

module load python/3.11.10
cd "$HOME/src/nncompile"
source .venv/bin/activate
mkdir -p experiments/exp_i/logs experiments/exp_i/results

METHOD="${1:-dmci}"
BUDGET="${2:-300}"
echo "=============================="
echo "Exp I comparison: method=$METHOD  budget=${BUDGET}s/method"
echo "Host:   $(hostname)"
echo "Python: $(python3 --version)"
echo "Start:  $(date)"
echo "=============================="

python3 -u -m experiments.exp_i.run_comparison \
    --method "$METHOD" --budget-s "$BUDGET" \
    --param-counts 1 2 3 4 --seeds 0 1 2 \
    --output-dir experiments/exp_i/results

echo ""
echo "End: $(date)"
