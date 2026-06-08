# FluZoo smoke / pilot artifacts (NOT the manuscript campaign)

These files are an **early small-scale pilot** of the FluZoo pipeline (a 4-program zoo,
1-generation / pop-2 evolution, 5-6 inner Adam steps). They are produced by
`run_all.py` and `aggregate.py` and are kept only as a provenance record of the pilot
configuration.

**They are NOT the Appendix-K (Experiment K) numbers.** The reported campaign lives one
level up in `results/`:

- `baseline_test.json`: the rigorous 300-step re-score (evolved val 0.01477 / test 0.01812,
  SEIR seed 0.01488 / 0.01805, SEIRS 0.01473 / 0.01856) that is the heart of the
  fitness-fidelity finding.
- `_pool_124_125/oe_pool_summary.json` and `oe_island_{0,1,2}/best/`: the OpenEvolve
  campaign (3 islands x 200 iterations; global best = 4-compartment, 11-parameter
  spatial-coupling structure, search-time val 0.0100).

Do not cite the numbers in this folder; they are the pilot, not the campaign.
