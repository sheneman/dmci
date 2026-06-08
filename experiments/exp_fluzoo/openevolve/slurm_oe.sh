#!/bin/bash
############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# slurm_oe.sh: FluZoo program evolution through OpenEvolve (AlphaEvolve-style outer loop) on the fast n128 node.
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

#SBATCH --job-name=fluzoo_oe
#SBATCH --partition=sheneman
#SBATCH --nodelist=n128
#SBATCH --cpus-per-task=56
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --output=experiments/exp_fluzoo/logs/oe_%j.out

# FluZoo program evolution through OpenEvolve (AlphaEvolve-style outer loop) on the fast n128 node.
# OpenEvolve's LLM ensemble (qwen3.6-35b + qwen3.6-27b, thinking on, max_tokens=65535 so the program
# is never truncated) diff-edits the Scheme model; each candidate is calibrated through the DMCI
# interpreter and scored on held-out forecast skill (oe_evaluator.py). MAP-Elites quality-diversity
# over [complexity, n_compartments]. Spawn workers (forced in run_oe.py).
set -euo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

cd /mnt/ceph/sheneman/src/nncompile
source .venv/bin/activate

mkdir -p experiments/exp_fluzoo/logs experiments/exp_fluzoo/results/oe
python3 -u -m experiments.exp_fluzoo.openevolve.run_oe \
    --models real --iterations 300 --workers 24 \
    --output-dir experiments/exp_fluzoo/results/oe "$@"
