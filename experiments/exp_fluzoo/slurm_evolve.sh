#!/bin/bash
############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# slurm_evolve.sh: LLM-guided evolutionary program search: each worker generates a child (LLM mutation/
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

#SBATCH --job-name=fluzoo_evolve
#SBATCH --partition=sheneman
#SBATCH --nodelist=n128
#SBATCH --cpus-per-task=56
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --output=experiments/exp_fluzoo/logs/evolve_%j.out

# LLM-guided evolutionary program search: each worker generates a child (LLM mutation/
# crossover) and scores its held-out forecast skill through the DMCI interpreter. Generation
# hits campus MindRouter (qwen) + OpenAI (GPT-5.5); scoring is CPU-bound. The model fit is
# season-batched (one shared-trajectory walk over all training seasons), so per-program cost
# is dominated by the held-out forecast scoring.
set -euo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

cd /mnt/ceph/sheneman/src/nncompile
source .venv/bin/activate

mkdir -p experiments/exp_fluzoo/logs
python3 -u -m experiments.exp_fluzoo.evolve \
    --generations 6 --pop 24 --elite 6 \
    --models qwen35,qwen27,gpt55 --workers 24 \
    --seeds 0 --adam-iters 80 --refit-iters 4 --origin-stride 10 \
    --child-timeout 1200 "$@"
