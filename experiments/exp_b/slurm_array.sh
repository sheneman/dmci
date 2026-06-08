#!/bin/bash
############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# slurm_array.sh: Experiment B SLURM array (hand-authored reference programs, path A)
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

#SBATCH --job-name=exp_b
#SBATCH --partition=eight
#SBATCH --array=0-3
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=experiments/exp_b/logs/exp_b_%A_%a.out
#SBATCH --error=experiments/exp_b/logs/exp_b_%A_%a.err

cd "$HOME/src/nncompile"
source .venv/bin/activate

mkdir -p experiments/exp_b/logs experiments/exp_b/results

case $SLURM_ARRAY_TASK_ID in
    0) MODELS="M01_coulomb M02_beer_lambert M03_michaelis_menten M04_arrhenius" ;;
    1) MODELS="M05_hookes_spring M06_logistic_growth M07_power_law M08_euler_ode" ;;
    2) MODELS="M09_taylor_exp M10_smooth_activation M11_recursive_filter M12_newton_sqrt" ;;
    3) MODELS="M13_composed_transforms M14_anomaly_scorer M15_horner_eval" ;;
esac

echo "Array task $SLURM_ARRAY_TASK_ID: models=$MODELS"

python3 -u -m experiments.exp_b.run_all \
    --models $MODELS \
    --output-dir experiments/exp_b/results \
    --skip-existing
