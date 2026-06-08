#!/bin/bash
############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# slurm_gate.sh: LIM-ENSO numerical GO/NO-GO gate. CPU / interpreter-bound on purpose: the LIM Kalman
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

#SBATCH --job-name=lim_gate
#SBATCH --partition=sheneman
#SBATCH --nodelist=n128
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=experiments/exp_lim_enso/logs/%x_%j.out
#SBATCH --error=experiments/exp_lim_enso/logs/%x_%j.err

# LIM-ENSO numerical GO/NO-GO gate. CPU / interpreter-bound on purpose: the LIM Kalman
# filter folds through the DMCI meta-circular interpreter as a scalar Python walk over the
# tiny tagged-tensor heap, so the GPU is SLOWER here. No GPU requested.
#
# uv .venv is self-contained on n128 (no conda / no module load -- see MEMORY/exp_i).
# Run by module so the package imports resolve from the repo root.
#
# Optionally gate a single dimension:   sbatch experiments/exp_lim_enso/slurm_gate.sh 10
# (omit the arg to gate the full config.D_list = [6, 10, 15, 20]).

cd /mnt/ceph/sheneman/src/nncompile
source .venv/bin/activate

mkdir -p experiments/exp_lim_enso/logs

echo "=============================="
echo "LIM-ENSO numerical gate"
echo "Host:   $(hostname)"
echo "Python: $(python3 --version)"
echo "Start:  $(date)"
echo "=============================="

if [ -n "$1" ]; then
    python3 -u -m experiments.exp_lim_enso.gate --D "$1"
else
    python3 -u -m experiments.exp_lim_enso.gate
fi

echo ""
echo "End: $(date)"
echo "Verdicts in experiments/exp_lim_enso/gate_{D}.json"
