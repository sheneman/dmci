#!/bin/bash
############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# slurm_parallel.sh: LIM-ENSO flagship sweep, run 16-way with PROCESS-LEVEL parallelism over independent fit
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

#SBATCH --job-name=lim_parallel
#SBATCH --partition=sheneman
#SBATCH --nodelist=n128
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=experiments/exp_lim_enso/logs/%x_%j.out
#SBATCH --error=experiments/exp_lim_enso/logs/%x_%j.err

# LIM-ENSO flagship sweep, run 16-way with PROCESS-LEVEL parallelism over independent fit
# cells (method x D x structure x seed). The DMCI interpreter is single-threaded Python
# (GIL-bound) so a single fit cannot use multiple cores -- the win is one worker process per
# core, each optimizing its OWN parameter vector and writing its OWN result. Parallel results
# are per-seed-IDENTICAL to the serial loop (each cell is deterministic from its seed).
#
# CPU / interpreter-bound on purpose: the LIM Kalman filter folds through the DMCI meta-circular
# interpreter as a scalar Python walk over the tiny tagged-tensor heap, so the GPU is SLOWER
# here. No GPU requested. n128 (partition 'sheneman') = 64 cores / 128 GB; we take 16 cores.
#
# NO thread oversubscription: 16 workers x 1 thread = 16 cores (NOT 16 x 64). The per-thread
# BLAS caps below are exported BEFORE python starts; each worker also calls
# torch.set_num_threads(1) at startup. spawn start method + a top-level picklable worker keep
# torch + multiprocessing deadlock-free.
#
# uv .venv is self-contained on n128 (no conda / no module load -- see MEMORY/exp_i). Run by
# module so the package imports resolve from the repo root.
#
# Pass-through args go to run_all (e.g. --skip-existing, --headline-D 10, --methods dmci_adam):
#   sbatch experiments/exp_lim_enso/slurm_parallel.sh --skip-existing
#   sbatch experiments/exp_lim_enso/slurm_parallel.sh --headline-D 10 --methods dmci_adam

# --- single-threaded BLAS per worker (export BEFORE python; the guardrail against oversub) ---
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

cd /mnt/ceph/sheneman/src/nncompile
source .venv/bin/activate

mkdir -p experiments/exp_lim_enso/logs experiments/exp_lim_enso/results

echo "=============================="
echo "LIM-ENSO parallel sweep (16-way process pool)"
echo "Host:   $(hostname)"
echo "Python: $(python3 --version)"
echo "Cores:  ${SLURM_CPUS_PER_TASK} (OMP=${OMP_NUM_THREADS} MKL=${MKL_NUM_THREADS})"
echo "Start:  $(date)"
echo "Args:   $@"
echo "=============================="

python3 -u -m experiments.exp_lim_enso.run_all --workers 16 "$@"

echo ""
echo "End: $(date)"
echo "Per-run artifacts in experiments/exp_lim_enso/results/<tag>.json/.csv"
echo "Portfolio summary in experiments/exp_lim_enso/results/run_all_summary.json"
