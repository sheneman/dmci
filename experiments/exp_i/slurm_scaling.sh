#!/bin/bash
############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# slurm_scaling.sh: Experiment I d-scaling sweep, parallelized as a SLURM job array: ONE task per
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

#SBATCH --job-name=exp_i_scaling
#SBATCH --partition=eight
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --array=0-8
#SBATCH --output=experiments/exp_i/logs/scaling_%A_%a.out
#SBATCH --error=experiments/exp_i/logs/scaling_%A_%a.err

# Experiment I d-scaling sweep, parallelized as a SLURM job array: ONE task per
# parameter count. Each task fits the FATES-anchored community-GPP model (batched DMCI
# gradient via reparam+L-BFGS+multistart, single-start Adam, and differential evolution)
# at an EQUAL per-fit wall-clock budget, over seeds 0,1,2.
#
#   d (free params) = 6 * n_pft        grid: d in {6,12,18,24,36,48,66,96,126}
#   n_pft           = 1,2,3,4,6,8,11,16,21
#
# This is an HONEST scaling characterization (how gradient-via-DMCI vs black-box accuracy
# scales with d at fixed budget), NOT a "DMCI wins" claim. CPU 'eight' partition (the
# scalar interpreter favors CPU). Args: [budget_s=300] [method=dmci|direct].

# The uv .venv is self-contained (Python 3.11.10 + torch/scipy/numpy/jax); no module
# load is needed (and the python/3.11.x modulefiles are not present on compute nodes).
cd "$HOME/src/nncompile"
source .venv/bin/activate
mkdir -p experiments/exp_i/logs experiments/exp_i/results/scaling

PFTS=(1 2 3 4 6 8 11 16 21)
P=${PFTS[$SLURM_ARRAY_TASK_ID]}
BUDGET="${1:-300}"
METHOD="${2:-dmci}"

echo "=============================================="
echo "Exp I scaling task ${SLURM_ARRAY_TASK_ID}: n_pft=$P  (d=$((6*P)) params)"
echo "budget=${BUDGET}s/method  method=$METHOD"
echo "Host: $(hostname)  Python: $(python3 --version)  Start: $(date)"
echo "=============================================="

python3 -u -m experiments.exp_i.run_comparison \
    --method "$METHOD" --budget-s "$BUDGET" \
    --param-counts "$P" --seeds 0 1 2 \
    --out-tag "pft${P}" \
    --output-dir experiments/exp_i/results/scaling

echo "End: $(date)"
