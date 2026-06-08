############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# compare_llm_rerun.py: Compare the LLM-program re-run (results_llm/) against the committed hand-authored run (results/): per-model...
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Compare the LLM-program re-run (results_llm/) against the committed hand-authored
run (results/): per-model DMCI-vs-direct equivalence on the LLM programs, and how the
DMCI numbers change vs the original. Reads only JSON (no torch needed)."""
import json, glob

MODELS = ["M01_coulomb","M02_beer_lambert","M03_michaelis_menten","M04_arrhenius",
          "M05_hookes_spring","M06_logistic_growth","M07_power_law","M08_euler_ode",
          "M09_taylor_exp","M10_smooth_activation","M11_recursive_filter","M12_newton_sqrt",
          "M13_composed_transforms","M14_anomaly_scorer","M15_horner_eval"]

def load(d, method, mdl):
    return [json.load(open(f)) for f in sorted(glob.glob(f"{d}/{method}_{mdl}_*.json"))]

def mean(xs):
    return sum(xs)/len(xs) if xs else float("nan")

NEW = "experiments/exp_b/results_llm"
OLD = "experiments/exp_b/results"

hdr = f"{'model':<20} | {'LLMdmci':>8} {'LLMdir':>7} {'epMatch':>7} {'maxLossDiff':>12} {'LLMdmciLoss':>12} | {'ORIGdmci':>8} {'ORIGdmciLoss':>12}"
print(hdr); print("-"*len(hdr))
for m in MODELS:
    dm = load(NEW, "dmci", m); di = load(NEW, "direct_compiled", m); om = load(OLD, "dmci", m)
    dconv = sum(r["converged"] for r in dm); iconv = sum(r["converged"] for r in di)
    em = sum(1 for a,b in zip(dm,di) if a["convergence_epoch"]==b["convergence_epoch"])
    mx = max(abs(a["final_loss"]-b["final_loss"]) for a,b in zip(dm,di))
    dml = mean([r["final_loss"] for r in dm])
    odc = f"{sum(r['converged'] for r in om)}/{len(om)}" if om else "-"
    odl = f"{mean([r['final_loss'] for r in om]):.4g}" if om else "-"
    print(f"{m:<20} | {dconv:>4}/{len(dm):<3} {iconv:>3}/{len(di):<3} {em:>4}/{len(dm):<2} {mx:>12.3g} {dml:>12.4g} | {odc:>8} {odl:>12}")

# Overall equivalence summary
allnew = [(m, load(NEW,"dmci",m), load(NEW,"direct_compiled",m)) for m in MODELS]
pairs = sum(min(len(a),len(b)) for _,a,b in allnew)
epm = sum(sum(1 for x,y in zip(a,b) if x["convergence_epoch"]==y["convergence_epoch"]) for _,a,b in allnew)
mxall = max(max((abs(x["final_loss"]-y["final_loss"]) for x,y in zip(a,b)), default=0) for _,a,b in allnew)
print(f"\nDMCI==direct on LLM programs: convergence-epoch match {epm}/{pairs} pairs; max final-loss diff {mxall:.3g}")
