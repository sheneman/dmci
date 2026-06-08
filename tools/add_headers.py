############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# add_headers.py: Insert the DMCI banner header into in-scope source files. Idempotent: a file already carrying the banner...
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################
"""Insert the DMCI banner header into in-scope source files.

Idempotent: a file already carrying the banner (detected by the author email) is skipped. The
per-file description is taken from the file's module docstring (Python), its leading comment
(shell), or a path-derived fallback (reported so it can be improved by hand).

    python3 tools/add_headers.py            # dry run: report what would change
    python3 tools/add_headers.py --apply    # write the headers in place
"""

from __future__ import annotations

import ast
import os
import sys

APPLY = "--apply" in sys.argv
UPDATE = "--update" in sys.argv   # re-derive the description line of files that already have the header
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
ROOTS = ["neural_compiler", "bootstrap", "experiments", "large_examples", "tests", "benchmarks",
         "examples", "tools"]
TOP_LEVEL_GLOBS = ["reproduce.sh"]
EXCLUDE_DIR = {"__pycache__", ".git", ".venv", "node_modules",
               "results", "oe_output", "bat_output", "checkpoints"}  # results/outputs are not source
SENTINEL = "sheneman@uidaho.edu"

# Curated descriptions for source files lacking a usable docstring/leading comment.
CURATED = {
    "bootstrap/compiler.scm": "The self-hosted meta-circular Scheme evaluator that DMCI compiles once into a differentiable interpreter",
    "large_examples/diffesm_s.scm": "DiffESM-S: a 97-node Earth-system model in Scheme, a production-scale batching benchmark (Experiment H)",
    "large_examples/diffsoc_s.scm": "DiffSoc-S: a 206-node urban political-economy simulator in Scheme, a production-scale batching benchmark (Experiment H)",
    "experiments/exp_a/config.py": "Experiment A configuration: the program suite, seeds, optimizer, and baseline settings",
    "experiments/exp_a/programs.py": "Experiment A program suite: the Scheme test programs and their ground-truth parameters",
    "experiments/exp_battery/openevolve/initial_program.py": "OpenEvolve seed: the smooth sqrt-t SEI battery-degradation model the structure search starts from",
    "experiments/exp_fluzoo/openevolve/initial_program.py": "OpenEvolve seed: the baseline SEIR-style influenza model the structure search starts from",
    # Terse SLURM submit wrappers with no original leading comment.
    "benchmarks/slurm_bench.sh": "SLURM submission for the DMCI benchmark suite",
    "experiments/exp_a/slurm_gradient_delta.sh": "Experiment A: SLURM job for the gradient-delta sub-experiment (autograd vs finite-difference)",
    "experiments/exp_b/slurm_array.sh": "Experiment B SLURM array (hand-authored reference programs, path A)",
    "experiments/exp_b/slurm_submit.sh": "Experiment B SLURM submission (hand-authored reference run, path A)",
    "experiments/exp_c/slurm_array.sh": "Experiment C SLURM array (recursive scientific models)",
    "experiments/exp_c/slurm_submit.sh": "Experiment C SLURM submission (recursive scientific models)",
    "experiments/exp_f/slurm_submit.sh": "Experiment F SLURM submission (LLM-in-the-loop model discovery)",
    "experiments/exp_g/slurm_submit.sh": "Experiment G SLURM submission (compositional modeling)",
    "experiments/exp_h/slurm_convergence.sh": "Experiment H: convergence-benchmark SLURM job",
    "experiments/exp_h/slurm_diffesm.sh": "Experiment H: DiffESM-S batching-benchmark SLURM job",
    "experiments/exp_h/slurm_diffsoc.sh": "Experiment H: DiffSoc-S batching-benchmark SLURM job",
    "experiments/exp_h/slurm_part_e.sh": "Experiment H: Part-E benchmark SLURM job",
}

PROJECT = ["DMCI: Compiling scheme into composable and",
           "      differentiable neural network representations"]
AFFIL = ["Luke Sheneman",
         "Research Computing and Data Services (RCDS)",
         "Institute for Interdisciplinary Data Sciences (IIDS)",
         "University of Idaho",
         "sheneman@uidaho.edu"]

# comment style per extension: (bar_char, blank_comment_line, body_prefix)
STYLE = {".py": ("#", "#", "# "), ".sh": ("#", "#", "# "), ".scm": (";", ";;", ";; ")}


def _clip(text: str | None) -> str | None:
    """Collapse wrapped whitespace to one line and trim at a word boundary (no mid-word cuts)."""
    t = " ".join((text or "").split())
    if not t:
        return None
    if len(t) <= 112:
        return t
    return t[:112].rsplit(" ", 1)[0] + "..."


def py_description(src: str) -> str | None:
    try:
        d = ast.get_docstring(ast.parse(src))
    except Exception:  # noqa: BLE001
        return None
    return _clip(d) if d else None


def sh_description(src: str) -> str | None:
    """First real description comment, skipping a shebang and an already-inserted banner block.

    The banner is delimited by two solid bar lines (>=10 '#'); when present we search after the
    second bar so an --update pass never re-reads the banner's own text as the description.
    #SBATCH directives and the synthesized '<file>:' line are not descriptions.
    """
    lines = src.splitlines()
    bars = [i for i, ln in enumerate(lines[:25]) if set(ln.strip()) == {"#"} and len(ln.strip()) >= 10]
    start = (bars[1] + 1) if len(bars) >= 2 else 1
    # Scan a generous window and skip code/#SBATCH lines so a description sitting below a
    # block of SBATCH directives or shell setup is still recovered (not just a top-of-file one).
    for ln in lines[start:start + 40]:
        s = ln.strip()
        if not s or not s.startswith("#"):
            continue
        body = s.lstrip("# ").strip()
        if len(body) > 4 and not body.startswith("SBATCH") and not body.endswith(":"):
            return _clip(body)
    return None


def fallback_description(relpath: str, top: str) -> str:
    base = os.path.basename(relpath)
    if base == "__init__.py":
        return f"Package initialization for the {os.path.dirname(relpath).replace(os.sep, '.')} module"
    return f"Source module for the {top} component (see {relpath})"


def make_header(ext: str, fname: str, desc: str) -> str:
    bar_char, blank, prefix = STYLE[ext]
    bar = bar_char * 60
    out = [bar, blank,
           prefix + PROJECT[0],
           prefix + PROJECT[1],
           blank,
           prefix + f"{fname}: {desc}",
           blank]
    out += [prefix + a for a in AFFIL]
    out += [blank, bar]
    return "\n".join(out) + "\n"


def insertion_index(lines: list[str], ext: str) -> int:
    i = 0
    if lines and lines[0].startswith("#!"):
        i = 1
    if ext == ".py" and i < len(lines) and "coding:" in lines[i]:
        i += 1
    return i


def process(path: str, relpath: str, top: str, ext: str, report: dict) -> None:
    with open(path, encoding="utf-8") as fh:
        src = fh.read()
    if ext == ".py":
        desc = py_description(src)
    elif ext == ".sh":
        desc = sh_description(src)
    else:
        desc = None
    if desc is None:
        desc = CURATED.get(relpath)
    if desc is None:
        desc = fallback_description(relpath, top)
        report["fallback"].append(relpath)
    base = os.path.basename(relpath)
    head = "\n".join(src.splitlines()[:25])
    if SENTINEL in head:
        if UPDATE:
            prefix = STYLE[ext][2]
            lines = src.splitlines()
            for i, ln in enumerate(lines[:25]):
                if ln.startswith(prefix + base + ":"):
                    lines[i] = prefix + f"{base}: {desc}"
                    break
            report["updated"].append(relpath)
            if APPLY:
                with open(path, "w", encoding="utf-8") as fh:
                    fh.write("\n".join(lines) + ("\n" if src.endswith("\n") else ""))
        else:
            report["skipped_have"].append(relpath)
        return
    header = make_header(ext, base, desc)
    lines = src.splitlines(keepends=True)
    idx = insertion_index(lines, ext)
    # ensure a blank line between header and following content
    tail = lines[idx:]
    sep = "" if (tail and tail[0].strip() == "") else "\n"
    new = "".join(lines[:idx]) + header + sep + "".join(tail)
    report["headed"].append(relpath)
    if APPLY:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(new)


def main() -> None:
    report = {"headed": [], "skipped_have": [], "fallback": [], "updated": []}
    targets = []
    for top in ROOTS:
        base = os.path.join(HERE, top)
        if not os.path.isdir(base):
            continue
        for dp, dns, fns in os.walk(base):
            dns[:] = [d for d in dns if d not in EXCLUDE_DIR and not d.endswith(".egg-info")
                      and not d.startswith(("bat_island", "oe_island"))]
            for fn in fns:
                ext = os.path.splitext(fn)[1]
                if ext not in STYLE or fn.endswith("_tmp.py"):
                    continue
                full = os.path.join(dp, fn)
                targets.append((full, os.path.relpath(full, HERE), top, ext))
    for g in TOP_LEVEL_GLOBS:
        full = os.path.join(HERE, g)
        if os.path.isfile(full):
            targets.append((full, g, "(top-level)", os.path.splitext(g)[1]))

    for full, rel, top, ext in sorted(targets):
        process(full, rel, top, ext, report)

    print(f"{'APPLIED' if APPLY else 'DRY RUN'}{' +UPDATE' if UPDATE else ''}: {len(targets)} in-scope files")
    print(f"  headed: {len(report['headed'])}")
    print(f"  updated (description refreshed): {len(report['updated'])}")
    print(f"  skipped (already have header): {len(report['skipped_have'])}")
    print(f"  used PATH-FALLBACK description ({len(report['fallback'])}, review these):")
    for r in report["fallback"]:
        print(f"      {r}")


if __name__ == "__main__":
    main()
