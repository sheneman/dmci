#!/bin/bash
############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# slurm_array.sh: 5 methods x 6 programs x 10 seeds = 300 runs
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

#SBATCH --job-name=exp_a
#SBATCH --partition=eight
#SBATCH --array=0-7
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=experiments/exp_a/logs/exp_a_%A_%a.out
#SBATCH --error=experiments/exp_a/logs/exp_a_%A_%a.err

# 5 methods x 6 programs x 10 seeds = 300 runs
# Split by method: fast methods (direct, handcoded_interp) share tasks,
# slow methods (compiled_interp, finite_diff, evolution_strategy) get more tasks

cd "$HOME/src/nncompile"
source .venv/bin/activate

mkdir -p experiments/exp_a/logs experiments/exp_a/results

case $SLURM_ARRAY_TASK_ID in
    0) METHODS="direct" PROGRAMS="P1_single_const,P2_multi_const,P3_recursive,P4_higher_order,P5_multi_function,P6_composed" ;;
    1) METHODS="handcoded_interp" PROGRAMS="P1_single_const,P2_multi_const,P3_recursive,P4_higher_order,P5_multi_function,P6_composed" ;;
    2) METHODS="compiled_interp" PROGRAMS="P1_single_const,P2_multi_const,P3_recursive" ;;
    3) METHODS="compiled_interp" PROGRAMS="P4_higher_order,P5_multi_function,P6_composed" ;;
    4) METHODS="finite_diff" PROGRAMS="P1_single_const,P2_multi_const,P3_recursive" ;;
    5) METHODS="finite_diff" PROGRAMS="P4_higher_order,P5_multi_function,P6_composed" ;;
    6) METHODS="evolution_strategy" PROGRAMS="P1_single_const,P2_multi_const,P3_recursive" ;;
    7) METHODS="evolution_strategy" PROGRAMS="P4_higher_order,P5_multi_function,P6_composed" ;;
esac

echo "Array task $SLURM_ARRAY_TASK_ID: methods=$METHODS programs=$PROGRAMS"

python3 -u -m experiments.exp_a.run_all \
    --methods "$METHODS" \
    --programs "$PROGRAMS" \
    --output-dir experiments/exp_a/results \
    --skip-existing \
    --no-ablations
