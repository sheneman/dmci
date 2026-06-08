#!/bin/bash
############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# slurm_oe_islands.sh: Multi-node OpenEvolve: each array task is an INDEPENDENT OpenEvolve run (its own MAP-Elites +
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

#SBATCH --job-name=fluzoo_oe_isl
#SBATCH --partition=eight-long
#SBATCH --array=1-10
#SBATCH --cpus-per-task=16
#SBATCH --mem=90G
#SBATCH --time=12:00:00
#SBATCH --output=experiments/exp_fluzoo/logs/oe_isl_%A_%a.out

# Multi-node OpenEvolve: each array task is an INDEPENDENT OpenEvolve run (its own MAP-Elites +
# internal islands, a distinct seed) on one eight-partition node -- "meta-islands". Combined with
# the fast n128 run, this restores the ~multi-node scale of the old hand-rolled island array while
# keeping OpenEvolve's quality-diversity. The eight nodes are ~2-3x slower than n128, so adam-iters
# are trimmed and the per-program eval cap raised so DMCI evals finish instead of timing out.
# Pool afterwards:  python3 -m experiments.exp_fluzoo.openevolve.pool_oe
set -euo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OE_ADAM_ITERS=45          # cheaper fit for the slower nodes
export OE_ORIGIN_STRIDE=12

cd /mnt/ceph/sheneman/src/nncompile
source .venv/bin/activate

SEED=${SLURM_ARRAY_TASK_ID}
OUT=experiments/exp_fluzoo/results/oe_island_${SEED}
mkdir -p experiments/exp_fluzoo/logs "${OUT}"

python3 -u -m experiments.exp_fluzoo.openevolve.run_oe \
    --models real --iterations 150 --workers 14 \
    --seed "${SEED}" --eval-timeout 2700 \
    --output-dir "${OUT}" "$@"
