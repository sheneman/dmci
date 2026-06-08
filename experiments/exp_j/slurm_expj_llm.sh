#!/bin/bash
############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# slurm_expj_llm.sh: Exp J — LLM-generated validation subset (external validity for the synthetic corpus).
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

#SBATCH --job-name=exp_j_llm
#SBATCH --partition=eight
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=experiments/exp_j/logs/expj_llm_%j.out
#SBATCH --error=experiments/exp_j/logs/expj_llm_%j.err

# Exp J — LLM-generated validation subset (external validity for the synthetic corpus).
# Programs are authored by MindRouter (qwen) and cached under experiments/exp_j/llm_cache/,
# then run through the same three arms (DMCI / B1-lambdify / B2-handport). Needs MindRouter
# egress (works from compute nodes) to POPULATE the cache on first run; reproducible offline
# afterwards. .env (MINDROUTER_API_KEY) must be present in the repo root on HPC.

module load python/3.11.10
cd "$HOME/src/nncompile"
source .venv/bin/activate
mkdir -p experiments/exp_j/logs experiments/exp_j/results experiments/exp_j/llm_cache

echo "Exp J (LLM)  host=$(hostname)  python=$(python3 --version)  start=$(date)"
python3 -u -m experiments.exp_j.run_expj_llm \
    --n-closed 200 --n-recursive 60 --workers 8 \
    --recover-sample 20 --recover-budget 30 \
    --output-dir experiments/exp_j/results
echo "end=$(date)"
