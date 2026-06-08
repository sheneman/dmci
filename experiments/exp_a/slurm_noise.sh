#!/bin/bash
############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# slurm_noise.sh: Experiment A noise-robustness sweep (review item R4): recover constants through the compiled
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

#SBATCH --job-name=exp_a_noise
#SBATCH --partition=eight
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=experiments/exp_a/logs/noise_%j.out
#SBATCH --error=experiments/exp_a/logs/noise_%j.err

# Experiment A noise-robustness sweep (review item R4): recover constants through the compiled
# self-hosted interpreter (DMCI) from data at several SNR levels; report relative parameter error
# vs noise. CPU partition (matches the paper's CPU narrative).

module load python/3.11.10
cd "$HOME/src/nncompile"
source .venv/bin/activate
mkdir -p experiments/exp_a/logs experiments/exp_a/results

echo "Exp A noise sweep  host=$(hostname)  python=$(python3 --version)  start=$(date)"
python3 -u -m experiments.exp_a.noise_sweep \
    --programs P1_single_const P2_multi_const P3_recursive P4_higher_order P5_multi_function \
    --sigmas 0.0 0.02 0.05 0.10 0.20 --seeds 5 \
    --max-epochs 1200 --patience 100 \
    --output experiments/exp_a/results/noise_sweep.json
echo "end=$(date)"
