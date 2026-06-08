#!/bin/bash
############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# slurm_sweep_array.sh: Multi-node sweep across the `eight` partition (~19 idle 16-core nodes): each array task
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

#SBATCH --job-name=fluzoo_sweep_arr
#SBATCH --partition=eight-long
#SBATCH --array=0-15
#SBATCH --cpus-per-task=16
#SBATCH --mem=90G
#SBATCH --time=08:00:00
#SBATCH --output=experiments/exp_fluzoo/logs/sweep_arr_%A_%a.out

# Multi-node sweep across the `eight` partition (~19 idle 16-core nodes): each array task
# scores a disjoint stride of the cached programs into a shared output dir. DMCI scoring is
# embarrassingly parallel across programs, so this gives near-linear speedup over one node.
#
# Workflow:
#   1) populate the zoo once (single node, hits the LLM endpoints):
#        python3 -m experiments.exp_fluzoo.llm_generate --n 270 --workers 24 --models qwen35,qwen27,gpt55
#   2) sbatch this array (16 tasks fan out over the eight partition)
#   3) merge once all tasks finish:
#        python3 -m experiments.exp_fluzoo.run_all --merge --output-dir experiments/exp_fluzoo/results/random
set -euo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

cd /mnt/ceph/sheneman/src/nncompile
source .venv/bin/activate
mkdir -p experiments/exp_fluzoo/logs experiments/exp_fluzoo/results/random

NSHARDS=${NSHARDS:-16}
python3 -u -m experiments.exp_fluzoo.run_all \
    --programs cache --workers 14 \
    --shard "${SLURM_ARRAY_TASK_ID}" --nshards "${NSHARDS}" \
    --output-dir experiments/exp_fluzoo/results/random \
    --seeds 0 --adam-iters 100 --refit-iters 5 --origin-stride 8 "$@"
