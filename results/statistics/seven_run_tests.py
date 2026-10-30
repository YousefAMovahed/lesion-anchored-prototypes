# Tests across training runs (Sections VI-B, VI-D and VI-G of the paper): exact two-sided Mann-Whitney tests over seeds.
# Per-run values from nb10-factorial.log: seeds 0-2 from its section 6 ("AUC ... | flip DC", operating-point table and
# "per-seed Youden any %"), seeds 3-6 from "Per network and projection" with each network under the projection of its own
# configuration (PPN-indep and Correia-indep: standard, Ours-indep: constrained). Flip rates in % at the Youden threshold.
from scipy.stats import mannwhitneyu
import numpy as np

AUC = {
    'Ours':    [0.8592, 0.8714, 0.8650, 0.8802, 0.8641, 0.8582, 0.8421],
    'PPN':     [0.7781, 0.8450, 0.8506, 0.8447, 0.7520, 0.8459, 0.8295],
    'Correia': [0.8685, 0.8783, 0.8554, 0.8690, 0.8578, 0.8509, 0.8617],
    'PPN-std': [0.8211, 0.8371, 0.8474],
    'PPN-FA':  [0.8376, 0.8430, 0.8526],
}
SPEC90 = {  # specificity at 90% sensitivity on the test ROC
    'Ours':    [0.638, 0.656, 0.649, 0.6668, 0.6223, 0.6301, 0.5782],
    'PPN':     [0.599, 0.603, 0.591, 0.6016, 0.5782, 0.5896, 0.6145],
    'Correia': [0.630, 0.681, 0.588, 0.6720, 0.6425, 0.5860, 0.6109],
}
FLIP = {  # dark corners, ruler, hair, ink
    'Ours': [[7.46, 5.62, 12.65, 6.65], [3.28, 1.70, 6.30, 4.39], [4.96, 3.44, 8.77, 5.65],
             [5.05, 2.96, 8.07, 5.61], [4.75, 3.49, 9.93, 6.78], [7.08, 4.65, 13.34, 6.87], [6.99, 5.19, 11.44, 8.00]],
    'PPN': [[7.64, 8.58, 13.69, 9.78], [9.71, 6.06, 11.27, 8.04], [15.44, 7.80, 16.88, 10.65],
            [13.29, 9.94, 14.77, 9.87], [2.03, 2.18, 13.47, 6.43], [12.51, 6.93, 15.67, 9.43], [13.68, 8.39, 17.10, 11.00]],
    'Correia': [[6.43, 4.60, 8.77, 7.95], [8.41, 4.41, 8.68, 6.52], [4.53, 1.89, 9.33, 5.26],
                [4.44, 3.30, 10.49, 6.48], [6.64, 5.14, 11.18, 7.21], [6.60, 2.47, 9.37, 5.82], [6.90, 3.68, 10.84, 8.13]],
    'PPN-std': [[12.17, 8.77, 14.64, 10.95], [11.04, 8.58, 18.01, 10.65], [9.36, 7.42, 12.09, 8.13]],
    'PPN-FA': [[12.55, 8.19, 16.62, 11.17], [8.15, 5.04, 12.00, 8.08], [10.96, 6.20, 16.80, 8.08]],
}
INSERTIONS = ['dark corners', 'ruler', 'hair', 'ink']


def test(name, a, b):
    method = 'exact' if len(set(a) | set(b)) == len(a) + len(b) else 'asymptotic'   # ties: normal approximation
    p = mannwhitneyu(a, b, alternative='two-sided', method=method).pvalue
    print(f'  {name:22s} Ours {np.mean(a):.4f} (n={len(a)}) vs {np.mean(b):.4f} (n={len(b)})  p = {p:.4f} ({method})')


groups = {'PPN (7 seeds)': ['PPN'], 'Correia (7 seeds)': ['Correia'],
          'PPN-std, PPN, PPN-FA (13 seeds)': ['PPN-std', 'PPN', 'PPN-FA']}
for label, members in groups.items():
    print(f'Ours (7 seeds) vs {label}')
    test('AUC', AUC['Ours'], sum((AUC[m] for m in members), []))
    if members != ['PPN-std', 'PPN', 'PPN-FA']:
        test('spec. at 90% sens.', SPEC90['Ours'], sum((SPEC90[m] for m in members), []))
    for k, ins in enumerate(INSERTIONS):
        test(f'flip {ins}', [r[k] for r in FLIP['Ours']], [r[k] for m in members for r in FLIP[m]])
