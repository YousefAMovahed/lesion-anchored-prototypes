"""Holm adjustment of the 16-test primary family (paper, Section V-C).

Raw two-sided p-values with lesion groups resampled (AUC: paired group bootstrap of the seed ensembles; artifact share
and flip rates: Wilcoxon on lesion-group means), copied from
  nb05-reviewer.log  (Ours vs. PPN-std, PPN, PPN-FA; "primary family" block) and
  nb06-mask-loss.log (Ours vs. the Correia et al. formulation).
Run: python holm_primary_family.py
"""
raw = {
    ('AUC', 'PPN-std'): 0.002, ('artifact share', 'PPN-std'): 1.63e-11,
    ('flip, dark corners', 'PPN-std'): 3.68e-27, ('flip, ruler', 'PPN-std'): 1.63e-16,
    ('AUC', 'PPN'): 0.014, ('artifact share', 'PPN'): 3.12e-06,
    ('flip, dark corners', 'PPN'): 7.41e-31, ('flip, ruler', 'PPN'): 7.88e-17,
    ('AUC', 'PPN-FA'): 0.007, ('artifact share', 'PPN-FA'): 1.07e-13,
    ('flip, dark corners', 'PPN-FA'): 1.69e-21, ('flip, ruler', 'PPN-FA'): 4.87e-08,
    ('AUC', 'Correia'): 0.851, ('artifact share', 'Correia'): 2.97e-30,
    ('flip, dark corners', 'Correia'): 0.00588, ('flip, ruler', 'Correia'): 0.55,
}
items = sorted(raw.items(), key=lambda kv: kv[1])
m, running = len(items), 0.0
for k, ((test, versus), p) in enumerate(items):
    running = max(running, min(1.0, (m - k) * p))      # Holm step-down, monotone
    print(f'{test:20s} Ours vs. {versus:8s} raw p = {p:9.3g}   Holm p = {running:9.3g}   {"significant" if running < 0.05 else "n.s."}')
