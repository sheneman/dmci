#!/bin/bash
############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# slurm_evolve_islands.sh: Island-model evolutionary search across the `eight` partition: each array task is an
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

#SBATCH --job-name=fluzoo_island
#SBATCH --partition=eight-long
#SBATCH --array=0-7
#SBATCH --cpus-per-task=16
#SBATCH --mem=90G
#SBATCH --time=12:00:00
#SBATCH --output=experiments/exp_fluzoo/logs/island_%A_%a.out

# Island-model evolutionary search across the `eight` partition: each array task is an
# INDEPENDENT evolve run with its own seed/population, on its own node. K islands give K x the
# evaluations in the same wall-clock AND broader structural coverage (diversity across islands).
# Pool them afterwards with the --merge-islands step below.
#
# Workflow:
#   1) sbatch this array (8 islands fan out over the eight partition; ~90 min each, in parallel)
#   2) merge once all islands finish (single node):
#        python3 -m experiments.exp_fluzoo.evolve --merge-islands \
#          experiments/exp_fluzoo/results/island_{0..7} \
#          --output-dir experiments/exp_fluzoo/results
set -euo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

cd /mnt/ceph/sheneman/src/nncompile
source .venv/bin/activate

SEED=${SLURM_ARRAY_TASK_ID}
OUT=experiments/exp_fluzoo/results/island_${SEED}
mkdir -p experiments/exp_fluzoo/logs "${OUT}"

# adam-iters trimmed to 60 (fit is the bottleneck) and a generous 1800s child cap so the timeout
# only catches genuine outliers, not merely-slow valid children. The forecast is now season-batched.
python3 -u -m experiments.exp_fluzoo.evolve \
    --generations 5 --pop 16 --elite 5 \
    --models qwen35,qwen27,gpt55 --workers 14 \
    --seeds 0 --adam-iters 60 --refit-iters 4 --origin-stride 10 \
    --child-timeout 1800 --seed "${SEED}" \
    --output-dir "${OUT}" "$@"
