"""
evalkit.py -- evaluation toolkit for the paper (NB02).
Classification metrics + DeLong + bootstrap, convex last-layer refit, prototype attribution
(lesion / artifact / background hits), Grad-CAM for the CNN, controlled counterfactual
artifact interventions (lesion pixels provably unchanged) and explanation-stability metrics.
"""
import os, json, math, numpy as np, pandas as pd, cv2, torch
import torch.nn.functional as F
from scipy.stats import norm, wilcoxon
from sklearn.metrics import roc_auc_score
import artproto as ap

OUT = ap.OUT  # 224


# ----------------------------------------------------------------------------- runs
def find_runs(files):
    by = {}
    for f in files:
        by.setdefault(os.path.dirname(f), set()).add(os.path.basename(f))
    runs = {}
    for d, fs in by.items():
        name = os.path.basename(d)
        if {"model.pt", "log.json"} <= fs and not name.endswith("_smoke"):
            if name not in runs or d.startswith("/kaggle/working"):
                runs[name] = d
    return dict(sorted(runs.items()))


def load_run(d, device):
    L = json.load(open(f"{d}/log.json"))
    cfg = L["cfg"]
    sd = torch.load(f"{d}/model.pt", map_location="cpu")
    m = ap.build_model(cfg, pretrained=False)
    if "features.conv0.weight" in sd:          # real DenseNet-121 checkpoints (no weight download needed)
        import torchvision
        m.features = torchvision.models.densenet121(weights=None).features
    m.load_state_dict(sd, strict=True)
    m = m.to(device).eval().to(memory_format=torch.channels_last)
    return m, cfg, L


# ----------------------------------------------------------------------------- forward passes
def _batches(n, bs):
    for s in range(0, n, bs):
        yield s, min(n, s + bs)


@torch.no_grad()
def forward(model, imgs, msks, idx, device, bs=128, keep_maps=False):
    """imgs/msks: arrays or memmaps (N,256,256,3)/(N,256,256); idx: positions to use.
    Returns dict prob (n,), act (n,P) and optionally sim (n,P,7,7) float16."""
    model.eval()
    probs, acts, sims = [], [], []
    for s, e in _batches(len(idx), bs):
        ii = idx[s:e]
        x, _, _ = ap.gpu_augment(torch.from_numpy(np.ascontiguousarray(imgs[ii])).to(device),
                                 torch.from_numpy(np.ascontiguousarray(msks[ii])).to(device), "eval")
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            out = model(x.contiguous(memory_format=torch.channels_last))
        probs.append(F.softmax(out["logits"].float(), 1)[:, 1].cpu())
        if "act" in out:
            acts.append(out["act"].float().cpu())
            if keep_maps:
                sims.append(out["sim"].half().cpu())
    r = {"prob": torch.cat(probs).numpy()}
    if acts:
        r["act"] = torch.cat(acts).numpy()
        if keep_maps:
            r["sim"] = torch.cat(sims).numpy()
    return r


@torch.no_grad()
def gradcam(model, imgs, msks, idx, device, bs=128):
    """Class-activation map of the melanoma logit. For a global-average-pooling + linear head,
    d(logit_mel)/d(f) = w_mel/(h*w) at every position, so Grad-CAM equals CAM (Zhou et al. 2016)
    up to a positive constant; this formulation needs no backward pass (low memory)."""
    model.eval()
    w = model.head.weight[1].float()
    cams = []
    for s, e in _batches(len(idx), bs):
        ii = idx[s:e]
        x, _, _ = ap.gpu_augment(torch.from_numpy(np.ascontiguousarray(imgs[ii])).to(device),
                                 torch.from_numpy(np.ascontiguousarray(msks[ii])).to(device), "eval")
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            f = F.relu(model.features(x.contiguous(memory_format=torch.channels_last)))
        cams.append(F.relu(torch.einsum("bchw,c->bhw", f.float(), w)).half().cpu())
    return torch.cat(cams).numpy()


# ----------------------------------------------------------------------------- convex last layer
def refit_last_layer(model, acts, y, class_w, l1=1e-4, steps=3000, lr=0.02):
    """Optimise the ProtoPNet last layer (linear map on prototype activations) to convergence:
    weighted CE + l1 on negative-class connections (as in Chen et al.). Deterministic, no val peeking."""
    dev = model.last.weight.device
    W = model.last.weight.detach().clone().float().requires_grad_(True)
    A = torch.as_tensor(acts, dtype=torch.float32, device=dev)
    Y = torch.as_tensor(y, dtype=torch.long, device=dev)
    cw = torch.as_tensor(class_w, dtype=torch.float32, device=dev)
    neg = (1 - model.class_identity).float()
    opt = torch.optim.Adam([W], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for _ in range(steps):
        loss = F.cross_entropy(A @ W.T, Y, weight=cw) + l1 * (W * neg).abs().sum()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    return W.detach(), float(loss.detach())


def probs_from_act(act, W):
    return torch.softmax(torch.as_tensor(act) @ torch.as_tensor(W).cpu().T, 1)[:, 1].numpy()


# ----------------------------------------------------------------------------- metrics
def youden_threshold(y, p):
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(y, p)
    k = int(np.argmax(tpr - fpr))
    return float(min(thr[k], 1.0))


def bin_metrics(y, p, thr):
    y = np.asarray(y).astype(int); pr = (np.asarray(p) >= thr).astype(int)
    tp = int(((pr == 1) & (y == 1)).sum()); tn = int(((pr == 0) & (y == 0)).sum())
    fp = int(((pr == 1) & (y == 0)).sum()); fn = int(((pr == 0) & (y == 1)).sum())
    sens = tp / max(tp + fn, 1); spec = tn / max(tn + fp, 1); ppv = tp / max(tp + fp, 1)
    auc = roc_auc_score(y, p) if 0 < y.sum() < len(y) else float("nan")
    return dict(AUC=auc, BAcc=(sens + spec) / 2, Sens=sens, Spec=spec, PPV=ppv,
                F1=2 * ppv * sens / max(ppv + sens, 1e-9), TP=tp, FN=fn, FP=fp, TN=tn)


def _midrank(x):
    J = np.argsort(x, kind="mergesort"); Z = x[J]; N = len(x); T = np.zeros(N); i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1); i = j
    T2 = np.empty(N); T2[J] = T + 1
    return T2


def delong(y, p1, p2):
    """Paired DeLong test (Sun & Xu 2014). Returns auc1, auc2, z, two-sided p."""
    y = np.asarray(y).astype(int)
    order = np.r_[np.where(y == 1)[0], np.where(y == 0)[0]]
    m = int(y.sum()); n = len(y) - m
    P = np.vstack([p1, p2])[:, order].astype(np.float64)
    tx = np.array([_midrank(r[:m]) for r in P]); ty = np.array([_midrank(r[m:]) for r in P])
    tz = np.array([_midrank(r) for r in P])
    aucs = tz[:, :m].sum(1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n; v10 = 1.0 - (tz[:, m:] - ty) / m
    S = np.cov(v01) / m + np.cov(v10) / n
    var = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    z = (aucs[0] - aucs[1]) / math.sqrt(max(var, 1e-12))
    return float(aucs[0]), float(aucs[1]), float(z), float(2 * norm.sf(abs(z)))


def bootstrap_auc_ci(y, p, groups=None, B=1000, seed=0):
    rs = np.random.RandomState(seed); y = np.asarray(y); p = np.asarray(p)
    if groups is None:
        groups = np.arange(len(y))
    ug, inv = np.unique(groups, return_inverse=True)
    members = [np.where(inv == g)[0] for g in range(len(ug))]
    vals = []
    for _ in range(B):
        pick = rs.randint(0, len(ug), len(ug))
        ii = np.concatenate([members[g] for g in pick])
        if 0 < y[ii].sum() < len(ii):
            vals.append(roc_auc_score(y[ii], p[ii]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


# ----------------------------------------------------------------------------- masks at network resolution
def masks224(msk_u8, device):
    """bit-field masks (b,256,256) -> dict of bool tensors (b,224,224) with the SAME resampling as the
    evaluation transform (bilinear, align_corners=False) and a 0.5 threshold."""
    m = torch.as_tensor(np.ascontiguousarray(msk_u8)).to(device).long()
    chans = {"L": (m & ap.BIT_L) > 0, "A": (m & ap.BIT_ART) > 0}
    for k, b in ap.ART_BITS.items():
        chans[k] = (m & b) > 0
    st = torch.stack([v.float() for v in chans.values()], 1)
    st = F.interpolate(st, size=(OUT, OUT), mode="bilinear", align_corners=False) >= 0.5
    return {k: st[:, i] for i, k in enumerate(chans)}


def hot_regions(maps7, size=OUT, q=0.95):
    """maps7 (b,k,7,7) float -> boolean top-(1-q) regions (b,k,size,size) after bilinear upsampling."""
    b, k = maps7.shape[:2]
    up = F.interpolate(maps7.float(), size=(size, size), mode="bilinear", align_corners=False)
    flat = up.reshape(b, k, -1)
    thr = torch.quantile(flat, q, dim=2, keepdim=True)
    return (flat >= thr).reshape(b, k, size, size), up


def region_stats(H, up, M):
    """H (b,k,S,S) bool hot regions; up the maps; M dict of (b,S,S) bool masks.
    Returns per-image means over the k maps: hit fraction of each mask + pointing (argmax in mask)."""
    hs = H.float().sum((2, 3)).clamp_min(1)
    arg = up.reshape(*up.shape[:2], -1).argmax(2)                     # (b,k)
    out = {}
    for name, m in M.items():
        mm = m[:, None].float()
        out[f"hit_{name}"] = ((H.float() * mm).sum((2, 3)) / hs).mean(1)
        out[f"point_{name}"] = torch.gather(m.reshape(m.shape[0], -1).float(), 1, arg).mean(1)
    return {k: v.cpu().numpy() for k, v in out.items()}


def attribution(maps7, acts, msk_u8, device, topk=3, q=0.95, bs=64):
    """Per-image attribution statistics for prototype models (top-k most activated prototypes) or the
    CNN (acts=None: maps7 is the (n,7,7) Grad-CAM)."""
    rows = {}
    n = len(maps7)
    for s, e in _batches(n, bs):
        mp = torch.as_tensor(maps7[s:e]).to(device).float()
        if acts is None:
            sel = mp[:, None]
        else:
            top = torch.as_tensor(np.argsort(-acts[s:e], 1)[:, :topk].copy()).to(device)
            sel = torch.gather(mp, 1, top[:, :, None, None].expand(-1, -1, mp.shape[2], mp.shape[3]))
        H, up = hot_regions(sel, q=q)
        M = masks224(msk_u8[s:e], device)
        M["B"] = ~M["L"] & ~M["A"]
        st = region_stats(H, up, M)
        for k, v in st.items():
            rows.setdefault(k, []).append(v)
        for k in ["L", "A"] + list(ap.ART_BITS):
            rows.setdefault(f"area_{k}", []).append(M[k].float().mean((1, 2)).cpu().numpy())
    return pd.DataFrame({k: np.concatenate(v) for k, v in rows.items()})


# ----------------------------------------------------------------------------- counterfactual interventions
def _allowed(L, margin=4):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin + 1, 2 * margin + 1))
    return cv2.dilate(L.astype(np.uint8), k) == 0


def _compose(img, layer, allowed):
    changed = (layer != img).any(-1) & allowed
    out = img.copy(); out[changed] = layer[changed]
    return out, changed


def add_hair(img, L, rng):
    layer = img.copy()
    for _ in range(int(rng.integers(12, 30))):
        p0 = rng.uniform(0, 256, 2); a = rng.uniform(0, 2 * np.pi); ln = rng.uniform(60, 200)
        p2 = p0 + ln * np.array([np.cos(a), np.sin(a)]); p1 = (p0 + p2) / 2 + rng.normal(0, 25, 2)
        t = np.linspace(0, 1, 60)[:, None]
        pts = ((1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2).astype(np.int32)
        c = rng.uniform(15, 70)
        cv2.polylines(layer, [pts], False, (int(c * 1.15), int(c * 0.95), int(c * 0.8)),
                      int(rng.integers(1, 3)), cv2.LINE_AA)
    return _compose(img, layer, _allowed(L))


def add_ruler(img, L, rng):
    allowed = _allowed(L)
    for side in rng.permutation(["top", "bottom", "left", "right"]):
        layer = img.copy(); col = int(rng.uniform(35, 80)); ln = int(rng.uniform(150, 220))
        off = int(rng.uniform(10, 30)); s0 = (256 - ln) // 2
        canvas = np.zeros((256, 256), np.uint8)
        cv2.line(canvas, (s0, off), (s0 + ln, off), 255, 2, cv2.LINE_AA)
        for k, x in enumerate(range(s0, s0 + ln + 1, 7)):
            cv2.line(canvas, (x, off), (x, off + (12 if k % 5 == 0 else 6)), 255, 1, cv2.LINE_AA)
        if side == "bottom": canvas = canvas[::-1]
        elif side == "left": canvas = canvas.T.copy()
        elif side == "right": canvas = canvas.T[:, ::-1].copy()
        drawn = canvas > 0
        if (drawn & ~allowed).sum() > 0.02 * drawn.sum():
            continue                                    # would touch the lesion margin -> try another side
        a = (canvas.astype(np.float32) / 255)[..., None]
        layer = (img * (1 - a) + col * a).astype(np.uint8)
        return _compose(img, layer, allowed)
    return None, None


def add_dark_corners(img, L, rng):
    yy, xx = np.mgrid[:256, :256]
    r = np.hypot(xx - 127.5, yy - 127.5) / (127.5 * np.sqrt(2))
    r0 = rng.uniform(0.72, 0.85)
    a = np.clip((r - r0) / 0.06, 0, 1)[..., None]
    layer = (img * (1 - a) + rng.uniform(0, 15) * a).astype(np.uint8)
    return _compose(img, layer, _allowed(L))


def add_ink(img, L, rng):
    ys, xs = np.where(L > 0)
    if len(ys) == 0:
        return None, None
    cy, cx = ys.mean(), xs.mean()
    ry = (ys.max() - ys.min()) / 2 + rng.uniform(12, 25); rx = (xs.max() - xs.min()) / 2 + rng.uniform(12, 25)
    layer = img.copy(); col = (int(rng.uniform(100, 135)), int(rng.uniform(30, 55)), int(rng.uniform(130, 170)))
    start = rng.uniform(0, 360)
    for k in range(int(rng.integers(2, 5))):
        a0 = start + k * 90 + rng.uniform(-10, 10)
        cv2.ellipse(layer, (int(cx), int(cy)), (int(rx), int(ry)), 0, a0, a0 + rng.uniform(35, 70), col, 3, cv2.LINE_AA)
    return _compose(img, layer, _allowed(L))


def remove_by_inpaint(img, L, mask):
    m = (cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0) & _allowed(L, 2)
    if m.sum() == 0:
        return None, None
    inp = cv2.inpaint(img, m.astype(np.uint8), 3, cv2.INPAINT_TELEA)
    out = img.copy(); out[m] = inp[m]
    return out, m


def null_inpaint(img, L, art_mask, rng):
    """Control: inpaint a region of the same shape as the artifact mask, shifted onto clean skin."""
    for _ in range(10):
        dy, dx = rng.integers(-100, 100, 2)
        sh = np.roll(np.roll(art_mask, dy, 0), dx, 1) & ~art_mask
        if sh.sum() >= 0.5 * art_mask.sum():
            return remove_by_inpaint(img, L, sh)
    return None, None


INTERVENTIONS = ["add_hair", "add_ruler", "add_dark_corners", "add_ink", "remove_hair_ruler", "null_inpaint"]


def make_counterfactuals(IM, MS, idx, kind, seed=0):
    """Returns (cf images (n,256,256,3) uint8, changed masks (n,256,256) bool, ok flags) for test indices idx.
    Lesion pixels are verified to be bit-identical to the original."""
    n = len(idx)
    CF = np.zeros((n, 256, 256, 3), np.uint8); CH = np.zeros((n, 256, 256), bool); ok = np.zeros(n, bool)
    for k, i in enumerate(idx):
        img = np.array(IM[i]); m = np.array(MS[i]); L = (m & ap.BIT_L) > 0
        rng = np.random.default_rng([seed, int(i), INTERVENTIONS.index(kind)])
        if kind == "add_hair": out, ch = add_hair(img, L, rng)
        elif kind == "add_ruler": out, ch = add_ruler(img, L, rng)
        elif kind == "add_dark_corners": out, ch = add_dark_corners(img, L, rng)
        elif kind == "add_ink": out, ch = add_ink(img, L, rng)
        elif kind == "remove_hair_ruler":
            ts = (m & ap.BIT_TS) > 0
            out, ch = remove_by_inpaint(img, L, ts) if ts.mean() >= ap.ART_MIN_FRAC["TS"] else (None, None)
        elif kind == "null_inpaint":
            ts = (m & ap.BIT_TS) > 0
            out, ch = null_inpaint(img, L, ts | False, rng) if ts.mean() >= ap.ART_MIN_FRAC["TS"] else (None, None)
            if out is not None:
                ch = ch & ((m & ap.BIT_ART) == 0)
                out2 = img.copy(); out2[ch] = out[ch]; out = out2
        else:
            raise ValueError(kind)
        if out is None or ch.sum() == 0:
            CF[k] = img; continue
        assert np.array_equal(out[L], img[L]), f"lesion pixels changed for {kind} image {i}"
        CF[k] = out; CH[k] = ch; ok[k] = True
    return CF, CH, ok


def stability(sim0, act0, sim1, act1, changed256, msk_u8, device, topk=3, q=0.95, bs=64):
    """Explanation stability for prototype models: same top-k prototypes (chosen on the original image)
    before vs after the intervention."""
    rows = {}
    for s, e in _batches(len(sim0), bs):
        a0 = act0[s:e]; top = torch.as_tensor(np.argsort(-a0, 1)[:, :topk].copy()).to(device)
        g = lambda S: torch.gather(torch.as_tensor(S[s:e]).to(device).float(), 1,
                                   top[:, :, None, None].expand(-1, -1, S.shape[2], S.shape[3]))
        m0, m1 = g(sim0), g(sim1)
        H0, _ = hot_regions(m0, q=q); H1, _ = hot_regions(m1, q=q)
        M = masks224(msk_u8[s:e], device)
        ch = F.interpolate(torch.as_tensor(changed256[s:e]).to(device).float()[:, None], size=(OUT, OUT),
                           mode="bilinear", align_corners=False)[:, 0] >= 0.5
        inter = (H0 & H1).float().sum((2, 3)); uni = (H0 | H1).float().sum((2, 3)).clamp_min(1)
        L = M["L"][:, None].float()
        h1 = H1.float().sum((2, 3)).clamp_min(1); h0 = H0.float().sum((2, 3)).clamp_min(1)
        rows.setdefault("iou", []).append((inter / uni).mean(1).cpu().numpy())
        rows.setdefault("les_before", []).append(((H0.float() * L).sum((2, 3)) / h0).mean(1).cpu().numpy())
        rows.setdefault("les_after", []).append(((H1.float() * L).sum((2, 3)) / h1).mean(1).cpu().numpy())
        rows.setdefault("capture", []).append(((H1.float() * ch[:, None].float()).sum((2, 3)) / h1).mean(1).cpu().numpy())
        t1 = np.argsort(-act1[s:e], 1)[:, :topk]; t0 = np.argsort(-a0, 1)[:, :topk]
        rows.setdefault("top_jaccard", []).append(np.array([len(set(a) & set(b)) / len(set(a) | set(b)) for a, b in zip(t0, t1)]))
        x0 = m0[:, 0].reshape(m0.shape[0], -1); x1 = m1[:, 0].reshape(m1.shape[0], -1)
        r0 = x0.argsort(1).argsort(1).float(); r1 = x1.argsort(1).argsort(1).float()
        r0 = r0 - r0.mean(1, keepdim=True); r1 = r1 - r1.mean(1, keepdim=True)
        rows.setdefault("spearman_top1", []).append(((r0 * r1).sum(1) / (r0.norm(dim=1) * r1.norm(dim=1)).clamp_min(1e-9)).cpu().numpy())
    return pd.DataFrame({k: np.concatenate(v) for k, v in rows.items()})


def paired_wilcoxon(a, b):
    d = np.asarray(a) - np.asarray(b)
    d = d[np.isfinite(d)]
    if len(d) < 10 or np.allclose(d, 0):
        return float("nan")
    return float(wilcoxon(d).pvalue)
