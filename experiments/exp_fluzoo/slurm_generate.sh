#!/bin/bash
############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# slurm_generate.sh: FluZoo program generation: the LLM proposes the model zoo and the validity funnel screens
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

#SBATCH --job-name=fluzoo_gen
#SBATCH --partition=sheneman
#SBATCH --nodelist=n128
#SBATCH --cpus-per-task=24
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=experiments/exp_fluzoo/logs/generate_%j.out

# FluZoo program generation: the LLM proposes the model zoo and the validity funnel screens
# each proposal. Requires MindRouter (campus-only) -- run on a node with campus network access.
# Accepted programs are cached to experiments/exp_fluzoo/llm_cache/ (version-controlled), so
# the sweep can run anywhere afterwards.
set -euo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

cd /mnt/ceph/sheneman/src/nncompile
source .venv/bin/activate

mkdir -p experiments/exp_fluzoo/logs
# build data first if needed (public API): python3 -m experiments.exp_fluzoo.data.build_data
python3 -u -m experiments.exp_fluzoo.llm_generate --n 500 --workers 24 "$@"
