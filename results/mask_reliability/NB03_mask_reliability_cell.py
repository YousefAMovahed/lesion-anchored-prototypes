# ============================================================================================
# EXTRA ANALYSIS FOR THE REVIEW (mask reliability, leakage audit, data distribution)
# Paste this as ONE new cell at the END of the NB02 notebook (the one that has the training-notebook
# output as input). Run first the two %%writefile cells (artproto.py, evalkit.py) and then this cell.
# GPU T4 is used. Runtime about 15-20 min. Everything is printed and saved to /kaggle/working/extra.
# ============================================================================================
import os, json, time, numpy as np, pandas as pd, torch, torch.nn.functional as F, torchvision, warnings
import importlib, artproto as ap, evalkit as ek; importlib.reload(ap); importlib.reload(ek)
warnings.filterwarnings('ignore')
from sklearn.metrics import roc_auc_score
T0 = time.time(); dev = torch.device('cuda')
OUTX = '/kaggle/working/extra'; os.makedirs(OUTX, exist_ok=True)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40); pd.set_option('display.max_rows', 200)

FILES = ap.walk_files('/kaggle/input') + ap.walk_files('/kaggle/working') + ap.walk_files('/tmp/nb01out') + ap.walk_files('/tmp/train_output')
P1, P2 = ap.find_prep_dirs(FILES); PM = P2 or P1; RUNS = ek.find_runs(FILES)
assert P1 and len(RUNS) == 24, (P1, len(RUNS))
meta = pd.read_csv(f'{PM}/meta.csv'); IM = np.load(f'{P1}/images.npy', mmap_mode='r'); MS = np.load(f'{PM}/masks.npy', mmap_mode='r')
y = meta.label.values.astype(int)
IDX = {s: np.where(meta.split.values == s)[0] for s in ('train', 'val', 'test')}
assert {k: len(v) for k, v in IDX.items()} == {'train': 20552, 'val': 2459, 'test': 2320}

# ---------------------------------------------------------------- 1. data distribution
print('=' * 30, '1. DATA DISTRIBUTION', '=' * 30)
DX = [c for c in ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC'] if c in meta.columns]
meta['dx'] = meta[DX].values.argmax(1); meta['dx'] = meta['dx'].map(dict(enumerate(DX)))
t1 = pd.crosstab(meta.split, meta.dx, margins=True).reindex(['train', 'val', 'test', 'All'])[DX + ['All']]
print(t1.to_string()); t1.to_csv(f'{OUTX}/dist_split_dx.csv')
t2 = pd.crosstab([meta.source, meta.split], meta.label, margins=True); print('\n', t2.to_string()); t2.to_csv(f'{OUTX}/dist_source_split_label.csv')
t3 = pd.crosstab(meta.split, meta.lesion_src, margins=True); print('\nlesion-mask provenance per split (gt = expert HAM10000 mask, pred = DeepLabV3 prediction):\n', t3.to_string())
t3.to_csv(f'{OUTX}/mask_provenance_split.csv')
for s in ('train', 'val', 'test'):
    print(f'  {s}: lesion_valid {meta.lesion_valid.values[IDX[s]].mean():.4f} | artifact prevalence',
          {k: round(float(meta[f"art_{k}"].values[IDX[s]].mean()), 3) for k in ap.ART_BITS})

# ---------------------------------------------------------------- 2. leakage audit
print('\n' + '=' * 30, '2. LEAKAGE AUDIT', '=' * 30)
checks = {}
for a in ('train', 'val', 'test'):
    for b in ('train', 'val', 'test'):
        if a < b:
            checks[f'shared groups {a}/{b}'] = len(set(meta.group[meta.split == a]) & set(meta.group[meta.split == b]))
            la = set(meta.lesion_id[(meta.split == a) & meta.lesion_id.notna()]); lb = set(meta.lesion_id[(meta.split == b) & meta.lesion_id.notna()])
            checks[f'shared lesion_id {a}/{b}'] = len(la & lb)
            checks[f'shared image ids {a}/{b}'] = len(set(meta.image[meta.split == a]) & set(meta.image[meta.split == b]))
dup = pd.read_csv(f'{PM}/dup_candidates.csv')
v = dup[dup.verified]
checks['verified near-duplicate pairs split across sets'] = int((meta.split.values[v.i.values] != meta.split.values[v.j.values]).sum())
strict = dup[(dup.corr_full >= 0.97) & (dup.corr_centre >= 0.97)]
checks['near-identical pairs (corr>=0.97) total'] = len(strict)
checks['near-identical pairs split across sets'] = int((meta.split.values[strict.i.values] != meta.split.values[strict.j.values]).sum())
# the segmentation network was trained only on training-split HAM10000 masks
gt = meta.has_gt_mask.values.astype(bool)
checks['GT masks in train (segmentation training set)'] = int((gt & (meta.split.values == 'train')).sum())
checks['GT masks in val+test (held out from segmentation training)'] = int((gt & (meta.split.values != 'train')).sum())
for k, val in checks.items(): print(f'  {k}: {val}')
json.dump({k: int(v) for k, v in checks.items()}, open(f'{OUTX}/leakage_audit.json', 'w'), indent=1)
for e in ('Derm7pt-derm', 'Derm7pt-clinic', 'PAD-UFES-20'):
    f = f'{PM}/ext_{e}_meta.csv'
    if os.path.exists(f):
        em = pd.read_csv(f); print(f'  {e}: images {len(em)}, flagged possible ISIC-train near-duplicates {int(em.dup_with_isic_train.sum())}')

# ---------------------------------------------------------------- helpers
def load(d):
    L = json.load(open(f'{d}/log.json')); cfg = L['cfg']; sd = torch.load(f'{d}/model.pt', map_location='cpu')
    m = ap.build_model(cfg, pretrained=False); m.features = torchvision.models.densenet121(weights=None).features
    m.load_state_dict(sd, strict=True); return m.to(dev).eval().to(memory_format=torch.channels_last), cfg
@torch.no_grad()
def cam(m, A, M, idx, bs=128):
    w = m.head.weight[1].float(); out = []
    for s in range(0, len(idx), bs):
        x, _, _ = ap.gpu_augment(torch.from_numpy(np.ascontiguousarray(A[idx[s:s+bs]])).to(dev), torch.from_numpy(np.ascontiguousarray(M[idx[s:s+bs]])).to(dev), 'eval')
        with torch.autocast('cuda', dtype=torch.float16):
            f = F.relu(m.features(x.contiguous(memory_format=torch.channels_last)))
        out.append(F.relu(torch.einsum('bchw,c->bhw', f.float(), w)).half().cpu())
    return torch.cat(out).numpy()
def masks_mod(msk, delta):
    """lesion mask dilated (delta>0) or eroded (delta<0) by |delta| px at 256 px; artifacts never inside the lesion."""
    if delta == 0: return msk
    import cv2
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * abs(delta) + 1, 2 * abs(delta) + 1)); out = np.array(msk).copy()
    for i in range(len(out)):
        L = (out[i] & ap.BIT_L).astype(np.uint8)
        L2 = cv2.dilate(L, k) if delta > 0 else cv2.erode(L, k)
        art = out[i] & ap.BIT_ART
        if delta > 0: art = np.where(L2 > 0, 0, art).astype(np.uint8)
        out[i] = art | (L2 * ap.BIT_L)
    return out

MAIN = ['CNN', 'PPN-std', 'PPN', 'PPN-FA', 'Ours', 'Ours-noLes', 'Ours-noArt', 'Ours-stdPush']
cfg_of = lambda r: r.rsplit('_s', 1)[0]
test = IDX['test']; tv = meta.iloc[test]
keep = tv.lesion_valid.values.astype(bool)
prov = np.where(tv.lesion_src.values == 'gt', 'expert (HAM10000)', 'predicted (BCN+MSK)')
src = tv.source.values
MS_test = np.array(MS[test])
MS_var = {d: masks_mod(MS_test, d) for d in (-5, 0, 5)}

# ---------------------------------------------------------------- 3. inference (+ convex last layer), attribution by mask provenance
print('\n' + '=' * 30, '3. RESULTS BY MASK PROVENANCE / MASK PERTURBATION', '=' * 30)
n1 = y[IDX['train']].sum(); n0 = len(IDX['train']) - n1
CW = [len(IDX['train']) / (2 * n0), len(IDX['train']) / (2 * n1)]
rows_attr, rows_auc, PROB, WS = [], [], {}, {}
for run, d in RUNS.items():
    t = time.time(); m, cfg = load(d)
    if cfg['model'] == 'proto':
        otr = ek.forward(m, IM, MS, IDX['train'], dev)
        W, _ = ek.refit_last_layer(m, otr['act'], y[IDX['train']], CW)
        ot = ek.forward(m, IM, MS, test, dev, keep_maps=True)
        WS[run] = W.cpu().numpy(); p = ek.probs_from_act(ot['act'], WS[run]); maps, acts = ot['sim'], ot['act']
    else:
        p = ek.forward(m, IM, MS, test, dev)['prob']; maps, acts = cam(m, IM, MS, test), None
    PROB[run] = p
    for delta, MSd in MS_var.items():
        df = ek.attribution(maps[keep], None if acts is None else acts[keep], MSd[keep], dev)
        df['prov'] = prov[keep]
        for g, sub in [('all', df)] + list(df.groupby('prov')):
            has = sub.area_A > 0.005
            rows_attr.append(dict(run=run, config=cfg_of(run), mask_change_px=delta, subset=g, n=len(sub),
                                  hit_lesion=sub.hit_L.mean(), hit_artifact=sub.hit_A[has].mean(),
                                  enrichment=(sub.hit_A[has] / sub.area_A[has]).mean(), point_artifact=sub.point_A.mean()))
    for g in ['all'] + sorted(set(src)):
        sel = np.ones(len(test), bool) if g == 'all' else (src == g)
        rows_auc.append(dict(run=run, config=cfg_of(run), subset=g, n=int(sel.sum()), mel=int(y[test][sel].sum()),
                             AUC=roc_auc_score(y[test][sel], p[sel])))
    del m; torch.cuda.empty_cache(); print(f'  {run:16s} done {time.time()-t:.0f}s', flush=True)
A = pd.DataFrame(rows_attr); A.to_csv(f'{OUTX}/attribution_by_provenance_and_mask_change.csv', index=False)
B = pd.DataFrame(rows_auc); B.to_csv(f'{OUTX}/auc_by_source.csv', index=False)
order = {c: i for i, c in enumerate(MAIN)}
def summ(df, cols, by):
    g = df.groupby(by + ['config'])[cols]; s = g.mean().round(3).astype(str) + ' ± ' + g.std().round(3).astype(str)
    s = s.reset_index(); s['o'] = s.config.map(order); return s.sort_values(by + ['o']).drop(columns='o')
print('\n--- attribution by lesion-mask provenance (original masks) ---')
print(summ(A[A.mask_change_px == 0], ['hit_lesion', 'hit_artifact', 'enrichment', 'point_artifact'], ['subset']).to_string(index=False))
print('\n--- attribution under lesion-mask perturbation (all test images): -5 px eroded / 0 / +5 px dilated ---')
print(summ(A[A.subset == 'all'], ['hit_lesion', 'hit_artifact', 'enrichment', 'point_artifact'], ['mask_change_px']).to_string(index=False))
print('\n--- test AUC by acquisition source (classification does not use masks at inference) ---')
print(summ(B, ['AUC'], ['subset']).to_string(index=False))
print('  subset sizes:', B.drop_duplicates('subset').set_index('subset')[['n', 'mel']].to_dict('index'))

# ---------------------------------------------------------------- 4. counterfactual flips by mask provenance (main models)
print('\n' + '=' * 30, '4. COUNTERFACTUAL FLIPS BY MASK PROVENANCE', '=' * 30)
thr = {}
for run, d in RUNS.items():
    if cfg_of(run) not in ('CNN', 'PPN', 'PPN-FA', 'PPN-std', 'Ours'): continue
    m, cfg = load(d)
    W = WS.get(run)                                   # reuse the SAME last layer as in section 3
    pv = ek.forward(m, IM, MS, IDX['val'], dev)['prob'] if W is None else ek.probs_from_act(ek.forward(m, IM, MS, IDX['val'], dev)['act'], W)
    thr[run] = (ek.youden_threshold(y[IDX['val']], pv), W); del m; torch.cuda.empty_cache()
rows_cf = []
CF_IDX = test[keep]
for kind in ('add_dark_corners', 'add_ruler'):
    CF, CH, ok = ek.make_counterfactuals(IM, MS, CF_IDX, kind)
    sel = np.where(ok)[0]; Ms = np.array(MS[CF_IDX[sel]])
    for run, (th, W) in thr.items():
        m, cfg = load(RUNS[run])
        o = ek.forward(m, CF[sel], Ms, np.arange(len(sel)), dev)
        p1 = o['prob'] if W is None else ek.probs_from_act(o['act'], W)
        p0 = PROB[run][keep][sel]
        pv_ = prov[keep][sel]
        for g in ['all', 'expert (HAM10000)', 'predicted (BCN+MSK)']:
            s = np.ones(len(sel), bool) if g == 'all' else (pv_ == g)
            rows_cf.append(dict(run=run, config=cfg_of(run), intervention=kind, subset=g, n=int(s.sum()),
                                flip=float(((p0[s] >= th) != (p1[s] >= th)).mean()), abs_dp=float(np.abs(p1[s] - p0[s]).mean())))
        del m; torch.cuda.empty_cache()
    del CF, CH
C = pd.DataFrame(rows_cf); C.to_csv(f'{OUTX}/counterfactual_by_provenance.csv', index=False)
print(summ(C, ['flip', 'abs_dp'], ['intervention', 'subset']).to_string(index=False))
print('\nEXTRA ANALYSIS FINISHED in', round((time.time() - T0) / 60, 1), 'min; files in', OUTX)
