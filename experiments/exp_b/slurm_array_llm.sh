#!/bin/bash
############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# slurm_array_llm.sh: Experiment B re-run on the ACTUAL LLM-generated programs (path B): the cached
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

#SBATCH --job-name=exp_b_llm
#SBATCH --partition=eight
#SBATCH --array=0-3
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=experiments/exp_b/logs/exp_b_llm_%A_%a.out
#SBATCH --error=experiments/exp_b/logs/exp_b_llm_%A_%a.err

# Experiment B re-run on the ACTUAL LLM-generated programs (path B): the cached
# llm_cache/*.json programs are compiled via the uniform evaluator wrapper
# (experiments/exp_b/llm_sources.py) with no per-model edits. Requires the
# letrec support added to bootstrap/compiler.scm and the raised recursion limit
# in run_all.py. CPU partition; python/3.11.10 module matches the .venv.

cd "$HOME/src/nncompile"
module load python/3.11.10
source .venv/bin/activate

mkdir -p experiments/exp_b/logs experiments/exp_b/results_llm

# Spread the four slow recursive models (M08, M09, M11, M12) across the 4 tasks.
case $SLURM_ARRAY_TASK_ID in
    0) MODELS="M01_coulomb M02_beer_lambert M03_michaelis_menten M08_euler_ode" ;;
    1) MODELS="M04_arrhenius M05_hookes_spring M06_logistic_growth M09_taylor_exp" ;;
    2) MODELS="M07_power_law M10_smooth_activation M11_recursive_filter M13_composed_transforms" ;;
    3) MODELS="M12_newton_sqrt M14_anomaly_scorer M15_horner_eval" ;;
esac

echo "Array task $SLURM_ARRAY_TASK_ID: models=$MODELS (LLM-cache mode, all 4 methods)"

python3 -u -m experiments.exp_b.run_all \
    --use-llm-cache \
    --models $MODELS \
    --output-dir experiments/exp_b/results_llm \
    --skip-existing
