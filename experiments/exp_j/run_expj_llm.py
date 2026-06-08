############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# run_expj_llm.py: Exp J — LLM-generated validation subset. External-validity companion to `run_expj.py`: instead of the synthetic...
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Exp J — LLM-generated validation subset.

External-validity companion to `run_expj.py`: instead of the synthetic random-tree corpus,
the programs here are authored by MindRouter (qwen) and cached. We run the identical three arms
(DMCI / B1-lambdify / B2-handport) and report, split by family (closed-form vs recursive):
  - coverage per arm  (does real LLM output reproduce the synthetic coverage story?)
  - cumulative compile time + the one-compile-vs-N-compiles crossover
  - per-structure engineering (LOC proxy)
  - matched recovery error (shared scipy L-BFGS-B multi-start)

Honest scoping (documented in LLM_VALIDATION.md): closed-form programs run all three arms with an
independent Python ground truth; recursive programs use DMCI ground truth, B1 fails coverage
(lambdify cannot ingest recursion — the point), and B2 is costed by LOC but its JAX port is not
auto-emitted, so B2 recovery is reported only on the closed-form family.

Run on HPC (compute-node egress populates the cache, then it is reproducible offline):
  python -m experiments.exp_j.run_expj_llm --n-closed 200 --n-recursive 60
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .arms import dmci_setup, dmci_prepare, b1_prepare, b2_prepare, recover
from .llm_corpus import generate_llm_corpus

_ARMS = ["dmci", "b1", "b2"]


def _cumulative_curve(prep, one_time):
    """Running cumulative compile cost across programs (for the crossover plot)."""
    cum, total = [], one_time
    for p in prep:
        total += p["compile_s"]
        cum.append(total)
    return cum


def run(n_closed, n_recursive, recover_sample, recover_budget, seed, output_dir, workers=8):
    output_dir.mkdir(parents=True, exist_ok=True)
    dmci = dmci_setup()
    print(f"DMCI one-time interpreter compile: {dmci['one_time_compile_s']:.2f}s, "
          f".ncg = {dmci['ncg_bytes']/1024:.0f} KB")

    print(f"\nGenerating LLM corpus (cached, {workers} concurrent calls): "
          f"{n_closed} closed-form + {n_recursive} recursive ...")
    corpus = generate_llm_corpus(n_closed, n_recursive, dmci["interp"],
                                 seed=seed, max_workers=workers)
    n_cf = sum(1 for p in corpus if p.kind == "closed_form")
    n_rec = sum(1 for p in corpus if p.kind == "recursive")
    print(f"Built {len(corpus)} distinct LLM programs: {n_cf} closed-form, {n_rec} recursive")

    # Prepare every program through every arm.
    prep = {a: [] for a in _ARMS}
    for j, prog in enumerate(corpus):
        prep["dmci"].append(dmci_prepare(prog, dmci["interp"]))
        prep["b1"].append(b1_prepare(prog))
        prep["b2"].append(b2_prepare(prog))
        if (j + 1) % max(1, len(corpus) // 10) == 0:
            print(f"  prepared {j+1}/{len(corpus)}")

    N = len(corpus)
    one_time = dmci["one_time_compile_s"]

    def arm_summary(a, idxs):
        pp = [prep[a][i] for i in idxs]
        ot = one_time if a == "dmci" else 0.0
        n = len(pp) or 1
        return {
            "n": len(pp),
            "cumulative_compile_s": ot + sum(p["compile_s"] for p in pp),
            "cumulative_eng_loc": sum(p["eng_loc"] for p in pp),
            "coverage": sum(1 for p in pp if p["covered"]) / n,
        }

    idx_all = list(range(N))
    idx_cf = [i for i, p in enumerate(corpus) if p.kind == "closed_form"]
    idx_rec = [i for i, p in enumerate(corpus) if p.kind == "recursive"]

    out = {
        "dmci_one_time_compile_s": one_time, "ncg_bytes": dmci["ncg_bytes"],
        "n_total": N, "n_closed_form": n_cf, "n_recursive": n_rec,
        "by_family": {
            "all": {a: arm_summary(a, idx_all) for a in _ARMS},
            "closed_form": {a: arm_summary(a, idx_cf) for a in _ARMS},
            "recursive": {a: arm_summary(a, idx_rec) for a in _ARMS},
        },
        # cumulative compile across all programs in corpus order — the crossover curve
        "cumulative_compile": {a: _cumulative_curve(prep[a], one_time if a == "dmci" else 0.0)
                               for a in _ARMS},
        "programs": [{"pid": p.pid, "kind": p.kind, "scheme": p.scheme,
                      "param_names": p.param_names, "port_loc": p.port_loc,
                      "dmci_covered": prep["dmci"][i]["covered"],
                      "b1_covered": prep["b1"][i]["covered"]}
                     for i, p in enumerate(corpus)],
        "recovery": [],
    }

    # B-arm cumulative crosses DMCI's flat one-time cost at this many programs:
    for a in ("b1", "b2"):
        cum = out["cumulative_compile"][a]
        cross = next((k + 1 for k, v in enumerate(cum) if v > one_time), None)
        out["by_family"]["all"][a]["crossover_N"] = cross

    print(f"\n=== coverage (fraction evaluable with no human intervention) ===")
    for fam in ("all", "closed_form", "recursive"):
        d = out["by_family"][fam]
        print(f"  {fam:11s}: DMCI={d['dmci']['coverage']:.2f} "
              f"B1={d['b1']['coverage']:.2f} B2={d['b2']['coverage']:.2f}  (n={d['dmci']['n']})")
    da = out["by_family"]["all"]
    print(f"\n=== cumulative compile over {N} real LLM programs ===")
    print(f"  DMCI={da['dmci']['cumulative_compile_s']:.2f}s (one compile, flat)  "
          f"B1={da['b1']['cumulative_compile_s']:.2f}s  B2={da['b2']['cumulative_compile_s']:.2f}s")
    print(f"  crossover (B-arm cumulative exceeds DMCI's one-time): "
          f"B1 @ N={da['b1'].get('crossover_N')}  B2 @ N={da['b2'].get('crossover_N')}")
    print(f"  engineering LOC: DMCI={da['dmci']['cumulative_eng_loc']} "
          f"B2={da['b2']['cumulative_eng_loc']}")

    # Matched recovery on a subsample, balanced across families.
    sample = ([p for p in corpus if p.kind == "closed_form"][:recover_sample]
              + [p for p in corpus if p.kind == "recursive"][:recover_sample])
    for prog in sample:
        rec = {"pid": prog.pid, "kind": prog.kind, "arms": {}}
        for a in _ARMS:
            if a == "b1" and prog.kind == "recursive":
                rec["arms"][a] = {"covered": False}
                continue
            if a == "b2" and prog.jax_forward is None:
                rec["arms"][a] = {"covered": True, "note": "port costed (LOC), recovery not auto-emitted"}
                continue
            mse, rel = recover(a, prog, interp=dmci["interp"],
                               time_budget=recover_budget, seed=0)
            rec["arms"][a] = {"covered": True, "best_mse": mse, "param_rel_error": rel}
        out["recovery"].append(rec)

    res_path = output_dir / "expj_llm_results.json"
    res_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nSaved: {res_path}")
    _recovery_summary(out["recovery"])
    return out


def _recovery_summary(recovery):
    if not recovery:
        return
    agg = {}
    for r in recovery:
        for a, d in r["arms"].items():
            if d.get("covered") and "param_rel_error" in d:
                agg.setdefault((r["kind"], a), []).append(d["param_rel_error"])
    print("\n=== matched recovery: mean param-rel-error by (kind, arm) ===")
    for (kind, a), v in sorted(agg.items()):
        print(f"  {kind:11s} {a:5s}: {sum(v)/len(v):.3e}  (n={len(v)})")


def main():
    ap = argparse.ArgumentParser(description="Exp J: LLM-generated validation subset")
    ap.add_argument("--n-closed", type=int, default=200)
    ap.add_argument("--n-recursive", type=int, default=60)
    ap.add_argument("--recover-sample", type=int, default=20)
    ap.add_argument("--recover-budget", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8, help="concurrent MindRouter calls")
    ap.add_argument("--output-dir", type=Path, default=Path("experiments/exp_j/results"))
    args = ap.parse_args()
    sys.setrecursionlimit(5000)
    run(args.n_closed, args.n_recursive, args.recover_sample, args.recover_budget,
        args.seed, args.output_dir, workers=args.workers)


if __name__ == "__main__":
    main()
