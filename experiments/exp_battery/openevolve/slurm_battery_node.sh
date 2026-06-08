#!/bin/bash
############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# slurm_battery_node.sh: One OpenEvolve battery meta-island on one node. Partition/nodelist/job-name passed on the sbatch
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

#SBATCH --cpus-per-task=24
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH --output=experiments/exp_battery/logs/bat_%x_%j.out

# One OpenEvolve battery meta-island on one node. Partition/nodelist/job-name passed on the sbatch
# command line so the SAME script runs across partitions (sheneman/n128, borowiec/n124, vasdekis/n125).
#   sbatch --partition=sheneman --nodelist=n128 --job-name=bat_n128 slurm_battery_node.sh 0 <out>
set -euo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd /mnt/ceph/sheneman/src/nncompile
source .venv/bin/activate
SEED=${1:?seed required}
OUT=${2:?output dir required}
WORKERS=${3:-24}
mkdir -p experiments/exp_battery/logs "${OUT}"
python3 -u -m experiments.exp_battery.openevolve.run_battery \
    --models real --iterations 200 --workers "${WORKERS}" \
    --seed "${SEED}" --eval-timeout 1800 --output-dir "${OUT}"
