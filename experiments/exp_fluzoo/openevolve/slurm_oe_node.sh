#!/bin/bash
############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# slurm_oe_node.sh: One OpenEvolve meta-island on one node. Partition/nodelist/job-name are passed on the sbatch
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
#SBATCH --output=experiments/exp_fluzoo/logs/oe_%x_%j.out

# One OpenEvolve meta-island on one node. Partition/nodelist/job-name are passed on the sbatch
# command line so the SAME script runs across partitions (sheneman/n128, borowiec/n124,
# vasdekis/n125). Args: $1 = seed (distinct per island), $2 = output dir.
#   sbatch --partition=sheneman --nodelist=n128 --job-name=oe_n128 slurm_oe_node.sh 0 <out>
set -euo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

cd /mnt/ceph/sheneman/src/nncompile
source .venv/bin/activate

SEED=${1:?seed required}
OUT=${2:?output dir required}
WORKERS=${3:-24}
mkdir -p experiments/exp_fluzoo/logs "${OUT}"

python3 -u -m experiments.exp_fluzoo.openevolve.run_oe \
    --models real --iterations 200 --workers "${WORKERS}" \
    --seed "${SEED}" --eval-timeout 1800 \
    --output-dir "${OUT}"
