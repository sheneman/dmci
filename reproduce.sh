#!/usr/bin/env bash
############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# reproduce.sh: One-command reproduction driver for "Compile Once, Differentiate Everywhere" (DMCI).
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

# One-command reproduction driver for "Compile Once, Differentiate Everywhere" (DMCI).
#
#   ./reproduce.sh           # quick verification (minutes): install + full test suite
#   ./reproduce.sh full      # print the HPC batch commands for the paper's experiments
#
# The quick path verifies the load-bearing claims that do not need a cluster: the self-hosted
# interpreter compiles and runs, gradients flow through it (DMCI == direct compilation), batched
# evaluation matches sequential bit-for-bit, and .ncg serialization round-trips. The full
# experiments (timings, scaling, LLM generation) are SLURM jobs on a CPU/GPU cluster.
set -euo pipefail
cd "$(dirname "$0")"

MODE="${1:-quick}"

if [ "$MODE" = "full" ]; then
  cat <<'EOF'
Full experiments are SLURM batch jobs. From a checkout on the cluster, activate the
self-contained project venv (no module system needed) and install the package:

  source .venv/bin/activate && pip install -e .

Main-paper experiments (Sections 4.1-4.5):

  sbatch experiments/exp_a/slurm_submit.sh             # Exp A: gradient fidelity / constant learning (+ baselines)
  sbatch experiments/exp_b/slurm_array_llm.sh          # Exp B: LLM-generated programs -> results_llm/ (the REPORTED run)
  sbatch experiments/exp_c/slurm_submit.sh             # Exp C: compiled recursive scientific models
  sbatch experiments/exp_h/slurm_submit.sh             # Exp H: batched throughput / scaling (GPU)
  bash   experiments/exp_battery/openevolve/run_real_islands.sh   # Exp L: battery co-search (real Severson; needs data/, see exp_battery/data/metadata.json)
  sbatch experiments/exp_battery/openevolve/slurm_rescore.sh experiments.exp_battery.openevolve.rescore_real

Appendix experiments (D-K):

  sbatch experiments/exp_d/slurm_submit.sh             # Exp D: structural search cost
  sbatch experiments/exp_e/slurm_submit_e1.sh          # Exp E: Gumbel-Softmax operator recovery
  sbatch experiments/exp_f/slurm_submit.sh             # Exp F: LLM-in-the-loop discovery
  sbatch experiments/exp_g/slurm_submit.sh             # Exp G: runtime program composition
  sbatch experiments/exp_h/slurm_vmap.sh               # Exp H: jax.vmap baseline comparison
  sbatch experiments/exp_i/slurm_scaling.sh            # Exp I: dimensional-scaling sweep
  sbatch experiments/exp_j/slurm_expj_llm.sh           # Exp J: program-space calibration vs compile-each
  sbatch experiments/exp_lim_enso/slurm_gate.sh        # LIM-ENSO: numerical GO/NO-GO gate (needs ERSSTv5, see exp_lim_enso/data/)
  sbatch experiments/exp_lim_enso/slurm_parallel.sh    # LIM-ENSO: Kalman-MLE flagship sweep
  sbatch experiments/exp_fluzoo/openevolve/slurm_oe_islands.sh   # Exp K (FluZoo): influenza co-search (needs ILINet, see exp_fluzoo/data/processed/metadata.json)

Real-data experiments (battery, LIM-ENSO, FluZoo) fetch public datasets first; each
data prerequisite and its provenance are documented in that experiment's data/metadata.json.
Results land in experiments/exp_*/results/ with a per-experiment README and MANIFEST.
The top-level REPRODUCIBILITY.md is the single entry point: per-experiment commands,
expected numbers, the artifact-to-table map, seeds, and hardware.
EOF
  exit 0
fi

echo "== DMCI reproduction (quick) =="
echo "[1/2] installing package (editable)"
pip install -q -e . >/dev/null

echo "[2/2] running the test suite (self-hosting, DMCI gradient flow, batched==sequential, serialization)"
python -m pytest tests/ -q

echo
echo "Quick verification passed. Run './reproduce.sh full' for the cluster experiment commands."
