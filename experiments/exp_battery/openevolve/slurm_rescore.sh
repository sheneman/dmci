#!/bin/bash
############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# slurm_rescore.sh: Rigorous re-score (synthetic or real) with a generous time limit. The first synthetic re-score
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

# Rigorous re-score (synthetic or real) with a generous time limit. The first synthetic re-score
# (job 5150851) was killed at a 2 h limit before finishing; the iters=300 evals (~1160 s each, 6
# programs x 2 splits) need well over that. Both rescore modules now also dump JSON incrementally,
# so even a kill preserves completed rows.
#   sbatch --partition=gpu-8 --nodelist=n003 --job-name=bat_resc2 slurm_rescore.sh experiments.exp_battery.openevolve.rescore
#   sbatch --partition=gpu-8 --nodelist=n003 --job-name=bat_rescreal slurm_rescore.sh experiments.exp_battery.openevolve.rescore_real
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=experiments/exp_battery/logs/%x_%j.out
set -euo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd /mnt/ceph/sheneman/src/nncompile
source .venv/bin/activate
MOD=${1:?module required}
python3 -u -m "$MOD"
