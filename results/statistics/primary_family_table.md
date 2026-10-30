# Primary family of 16 tests (Section V-C of the paper)

Ours against PPN-std, PPN, PPN-FA and the strongest mask-loss variant (Correia et al. formulation) on ISIC test,
seeds 0-2. Effect = Ours minus the other model. AUC: seed-ensemble AUC, 95% CI and p from a paired bootstrap over
lesion groups (2,000 resamples). Artifact share and flip rates: mean over the three seeds; p from the paired Wilcoxon
test on lesion-group means of the per-image values (averaged over seeds). Holm step-down adjustment over all 16 tests:
`holm_primary_family.py` (raw p from `nb05-reviewer.log` and `nb06-mask-loss.log`).

| Test | Versus | Ours | Other | Effect (Ours − other) [95% CI] | Raw p | Holm p |
|---|---|---|---|---|---|---|
| Ensemble AUC | PPN-std | 0.884 | 0.861 | +0.022 [0.009, 0.036] | 0.002 | 0.012 |
| Ensemble AUC | PPN | 0.884 | 0.866 | +0.018 [0.004, 0.032] | 0.014 | 0.042 |
| Ensemble AUC | PPN-FA | 0.884 | 0.866 | +0.017 [0.005, 0.031] | 0.007 | 0.029 |
| Ensemble AUC | Correia | 0.884 | 0.885 | −0.001 [−0.012, 0.011] | 0.851 | 1 (n.s.) |
| Artifact share of the hot region | PPN-std | 4.8% | 8.2% | −3.4 pts | 1.6e-11 | 1.5e-10 |
| Artifact share of the hot region | PPN | 4.8% | 6.6% | −1.8 pts | 3.1e-06 | 2.2e-05 |
| Artifact share of the hot region | PPN-FA | 4.8% | 10.2% | −5.4 pts | 1.1e-13 | 1.1e-12 |
| Artifact share of the hot region | Correia | 4.8% | 14.1% | −9.3 pts | 3.0e-30 | 4.5e-29 |
| Flip rate, inserted dark corners | PPN-std | 5.2% | 10.9% | −5.7 pts | 3.7e-27 | 5.2e-26 |
| Flip rate, inserted dark corners | PPN | 5.2% | 10.9% | −5.7 pts | 7.4e-31 | 1.2e-29 |
| Flip rate, inserted dark corners | PPN-FA | 5.2% | 10.6% | −5.4 pts | 1.7e-21 | 2.2e-20 |
| Flip rate, inserted dark corners | Correia | 5.2% | 6.5% | −1.3 pts | 0.0059 | 0.029 |
| Flip rate, inserted ruler | PPN-std | 3.6% | 8.3% | −4.7 pts | 1.6e-16 | 1.8e-15 |
| Flip rate, inserted ruler | PPN | 3.6% | 7.5% | −3.9 pts | 7.9e-17 | 9.5e-16 |
| Flip rate, inserted ruler | PPN-FA | 3.6% | 6.5% | −2.9 pts | 4.9e-08 | 3.9e-07 |
| Flip rate, inserted ruler | Correia | 3.6% | 3.6% | 0.0 pts | 0.55 | 1 (n.s.) |

The tests treat the trained networks as fixed (uncertainty of the test sample). Across training runs (seven seeds of
PPN, the Correia formulation and Ours; `seven_run_tests.py`), Ours differs from PPN in AUC and in all four flip rates
(p ≤ 0.026) but not from the Correia formulation (p ≥ 0.62); on seeds 3-6 the artifact-share and dark-corner advantages
over the Correia formulation do not replicate (`nb10-factorial.log`, paper Section VI-G).
