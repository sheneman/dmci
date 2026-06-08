#!/bin/bash
############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# slurm_pilot.sh: Exp I de-risking pilot. CPU partition on purpose: the interpreter is scalar with
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

#SBATCH --job-name=exp_i_pilot
#SBATCH --partition=eight
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --output=experiments/exp_i/logs/pilot_%j.out
#SBATCH --error=experiments/exp_i/logs/pilot_%j.err

# Exp I de-risking pilot. CPU partition on purpose: the interpreter is scalar with
# a Python per-datapoint loop and tiny tensors, so the GPU is SLOWER here (the paper
# shows CPU beating GPU on DiffESM-S). No GPU requested.

module load python/3.11.10
cd "$HOME/src/nncompile"
source .venv/bin/activate

mkdir -p experiments/exp_i/logs experiments/exp_i/results experiments/exp_i/data

echo "=============================="
echo "Exp I PILOT (go/no-go gate)"
echo "Host:   $(hostname)"
echo "Python: $(python3 --version)"
echo "Start:  $(date)"
echo "=============================="

python3 -u -m experiments.exp_i.run_pilot --n-pft 2 \
    --output-dir experiments/exp_i/results

echo ""
echo "End: $(date)"
echo "Verdict + per-seed results in experiments/exp_i/results/pilot_result.json"
