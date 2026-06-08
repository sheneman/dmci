#!/bin/bash
############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# slurm_sweep.sh: FluZoo calibration sweep: fit + held-out forecast scoring for every accepted program.
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

#SBATCH --job-name=fluzoo_sweep
#SBATCH --partition=sheneman
#SBATCH --nodelist=n128
#SBATCH --cpus-per-task=56
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --output=experiments/exp_fluzoo/logs/sweep_%j.out

# FluZoo calibration sweep: fit + held-out forecast scoring for every accepted program.
# DMCI is interpreter-bound (single-threaded Python), so parallelism is PROCESS-level over
# independent programs on CPU -- the RTX 4090 on n128 is not requested.
set -euo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

cd /mnt/ceph/sheneman/src/nncompile
source .venv/bin/activate

mkdir -p experiments/exp_fluzoo/logs
python3 -u -m experiments.exp_fluzoo.run_all --programs cache --workers 56 "$@"
