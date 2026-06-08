############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# config.py: Single source of truth for the LLM + DMCI Flu Model Zoo (FluZoo). FluZoo is the co-search flagship: a language...
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Single source of truth for the LLM + DMCI Flu Model Zoo (FluZoo).

FluZoo is the co-search flagship: a language model searches the DISCRETE space of
influenza model programs (compartmental / seasonal / observation / closure structures,
written as Scheme) while a single frozen DMCI interpreter searches each program's
CONTINUOUS parameters by reverse-mode gradient descent. Held-out forecast skill
selects programs. This module pins every knob so the generation, gate, calibration,
forecasting, and aggregation stages all agree.

Mirrors experiments/exp_lim_enso/config.py (the LIM-ENSO flagship) deliberately.
"""

from __future__ import annotations

from dataclasses import dataclass

# 11 spatial units: national + the 10 HHS regions, as Delphi Epidata `fluview`
# region codes. Row 0 is national; rows 1..10 are HHS-1..HHS-10. The observation
# matrix bound into every program is obs[T_weeks, 11] in exactly this column order.
REGIONS: tuple[str, ...] = (
    "nat", "hhs1", "hhs2", "hhs3", "hhs4", "hhs5",
    "hhs6", "hhs7", "hhs8", "hhs9", "hhs10",
)


@dataclass(frozen=True)
class FluZooConfig:
    # ---- data: ILINet weighted %ILI, MMWR weeks --------------------------
    regions: tuple[str, ...] = REGIONS
    n_regions: int = 11
    # An influenza season is MMWR epiweek 40 of `year` .. epiweek 39 of `year+1`;
    # we identify it by its START year. Splits per the FluZoo design (pandemic
    # 2020-21 excluded from primary evaluation, reported separately as drift).
    train_seasons: tuple[int, ...] = (2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017)
    val_seasons: tuple[int, ...] = (2018, 2021, 2022)
    test_seasons: tuple[int, ...] = (2023, 2024)
    pandemic_seasons: tuple[int, ...] = (2020,)   # stress test only, never in selection
    weeks_per_season: int = 52
    horizons: tuple[int, ...] = (1, 2, 3, 4)      # weeks-ahead forecast targets

    # ---- outer loop: LLM program search (discrete) -----------------------
    n_programs: int = 500
    llm_model: str = "qwen/qwen3.6-35b"           # single campus workhorse (MindRouter)
    gen_temperature: float = 0.9                  # diversity for the zoo
    max_repairs: int = 3                          # VALIDATE -> REPAIR budget per program

    # ---- inner loop: DMCI gradient calibration (continuous) --------------
    seeds: tuple[int, ...] = (0, 1, 2)            # multi-start for the portfolio fit
    lr: float = 0.05
    adam_iters: int = 300
    grad_clip: float = 10.0
    conv_tol: float = 1e-4
    obs_var_floor: float = 1e-6                   # float32 Gaussian-NLL stability lever

    # ---- interpreter evaluation caps -------------------------------------
    # Raised well above the LIM-ENSO defaults: a flu rollout over multiple seasons
    # (T up to ~8 seasons * 52 weeks) through the meta-circular interpreter issues
    # many trampoline bounces and conses ~linearly with interpreted steps.
    recursion_limit: int = 20000
    eval_max_iter: int = 4_000_000
    eval_max_depth: int = 4_000_000
    eval_max_heap: int = 12_000_000
    log_every: int = 25

    @property
    def EVAL_KW(self) -> dict:
        return dict(max_iter=self.eval_max_iter,
                    max_depth=self.eval_max_depth,
                    max_heap=self.eval_max_heap)

    @property
    def all_seasons(self) -> tuple[int, ...]:
        return tuple(sorted(set(self.train_seasons + self.val_seasons
                                + self.test_seasons + self.pandemic_seasons)))


@dataclass(frozen=True)
class GateThresholds:
    """Numerical GO/NO-GO levers for the de-risk gate and the validity funnel."""
    # forward parity of a HAND-WRITTEN reference program vs its numpy twin (the
    # LLM zoo has no per-program oracle; parity applies only to harness self-test).
    parity_rel: float = 3e-3
    # minimum summed parameter-gradient norm for a program to count as "gradients flow".
    grad_floor: float = 1e-8
    # DMCI float32 autograd vs float64 central finite-difference (reference programs).
    fd_rel: float = 5e-2
    fd_eps: float = 1e-3
    fd_n_probe: int = 6
    # a generated program's NLL must stay finite across this many random param draws
    # to pass the "stable rollout" funnel stage.
    rollout_probes: int = 4
    # reject programs with more than this many scalar parameters: a guard against the
    # mutation operator growing unboundedly complex (hence slow-to-score) programs.
    max_params: int = 50


DEFAULT = FluZooConfig()
GATE = GateThresholds()

# Placeholder token the generated/templated program uses for its rollout horizon;
# materialize() substitutes the integer T (number of weeks) before compilation so
# the loop bound is a data-independent literal and never a detected free variable.
HORIZON_TOKEN = "NWEEKS"
