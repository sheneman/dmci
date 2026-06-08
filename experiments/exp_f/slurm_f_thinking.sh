#!/bin/bash
############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# slurm_f_thinking.sh: Experiment F, thinking-mode re-run. 12 (target x seed) discovery runs execute concurrently
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

#SBATCH --job-name=exp_f_think
#SBATCH --partition=eight
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=experiments/exp_f/logs/exp_f_think_%j.out
#SBATCH --error=experiments/exp_f/logs/exp_f_think_%j.err

# Experiment F, thinking-mode re-run. 12 (target x seed) discovery runs execute concurrently
# (one process each). Reasoning ENABLED; max_completion_tokens=32768; batched DMCI fit.
# Providers selected via the SPECS env var (space-separated), e.g.:
#   sbatch --export=ALL,SPECS="qwen27b_think"               slurm_f_thinking.sh
#   sbatch --export=ALL,SPECS="gpt55_think"                 slurm_f_thinking.sh   # needs OPENAI_API_KEY in .env
#   sbatch --export=ALL,SPECS="qwen27b_think gpt55_think"   slurm_f_thinking.sh
# Needs network egress: MindRouter (qwen) and/or api.openai.com (gpt-5.5).

module load python/3.11.10
cd "$HOME/src/nncompile"
source .venv/bin/activate
mkdir -p experiments/exp_f/logs experiments/exp_f/results_qwen27b_think experiments/exp_f/results_gpt55_think

SPECS="${SPECS:-qwen27b_think}"
FITTER="${FITTER:-adam}"
TARGETS_ARG=""
[ -n "$TARGETS" ] && TARGETS_ARG="--targets $TARGETS"
echo "Exp F (thinking)  host=$(hostname)  python=$(python3 --version)  specs=[$SPECS]  fitter=$FITTER  targets=[${TARGETS:-all}]  start=$(date)"
python3 -u -m experiments.exp_f.run_f_thinking --specs $SPECS --workers 8 \
    --fitter $FITTER $TARGETS_ARG --output-root experiments/exp_f
echo "end=$(date)"
