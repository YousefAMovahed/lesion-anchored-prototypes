"""Holm check of the 12 main comparisons of the background, diversity and coverage analyses (paper, Section V-C).

These analyses are secondary; the check shows which of their main comparisons would survive a Holm correction.
Raw two-sided p-values, copied from
  nb11-mechanisms.log (lesion-only images: seed-ensemble AUC of Ours vs. PPN-std, PPN, PPN-FA and the IAIA-BL
                       formulation, paired bootstrap over lesion groups with 2,000 resamples) and
  nb13-context.log    (BG-Gap of the seed ensembles vs. PPN-std, PPN, PPN-FA, CNN and Ours-noLes, same bootstrap;
                       participation ratio and diagnostic coverage across the seven runs of Ours vs. those of PPN
                       and of the Correia et al. formulation, exact Mann-Whitney).
The participation ratio is entered once, as in the paper (every Ours run above every run of both models, the
same exact p against each; entering it twice gives 11 of 13). A bootstrap p printed as 0.000 is entered as 0.0005 (fewer than one of 2,000 resamples), which can only raise the
adjusted values.
Run: python holm_secondary_check.py
"""
raw = {
    ('lesion-only AUC', 'PPN-std'): 0.0005, ('lesion-only AUC', 'PPN'): 0.015,
    ('lesion-only AUC', 'PPN-FA'): 0.030, ('lesion-only AUC', 'IAIA-BL'): 0.030,
    ('BG-Gap', 'PPN-std'): 0.0005, ('BG-Gap', 'PPN'): 0.002, ('BG-Gap', 'PPN-FA'): 0.0005,
    ('BG-Gap', 'CNN'): 0.0005, ('BG-Gap', 'Ours-noLes'): 0.0005,
    ('participation ratio', 'PPN, Correia'): 0.0006,
    ('diagnostic coverage', 'PPN'): 0.0023, ('diagnostic coverage', 'Correia'): 0.0023,
}
items = sorted(raw.items(), key=lambda kv: kv[1])
m, running, kept = len(items), 0.0, 0
for k, ((test, versus), p) in enumerate(items):
    running = max(running, min(1.0, (m - k) * p))      # Holm step-down, monotone
    kept += running < 0.05
    print(f'{test:20s} Ours vs. {versus:12s} raw p = {p:7.4f}   Holm p = {running:6.4f}   {"significant" if running < 0.05 else "n.s."}')
print(f'{kept} of {m} comparisons significant after Holm correction')
