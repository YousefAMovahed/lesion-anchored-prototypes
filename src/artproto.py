"""
artproto.py -- shared library for the paper
"Artifact-aware prototype learning for skin-lesion classification".

Everything (data caching, lesion/artifact masks, models, losses, training,
push, evaluation, counterfactuals) lives here so that every notebook uses
exactly the same code.
"""
import os, re, json, glob, math, time, random, copy, traceback
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

IMG = 256          # cached resolution (square, as in the original project)
OUT = 224          # network input resolution
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

# bit layout of the uint8 mask array
BIT_L, BIT_DC, BIT_TS, BIT_SP, BIT_INK = 1, 2, 4, 8, 16
BIT_ART = BIT_DC | BIT_TS | BIT_SP | BIT_INK
ART_BITS = {"DC": BIT_DC, "TS": BIT_TS, "SP": BIT_SP, "INK": BIT_INK}
ART_LONG = {"DC": "dark corner / vignette", "TS": "thin dark structures (hair, ruler ticks)",
            "SP": "specular highlights / gel bubbles", "INK": "ink markings / coloured patches"}
DETECTOR_VERSION = 2
# image-level presence thresholds (fraction of image area)
ART_MIN_FRAC = {"DC": 0.01, "TS": 0.003, "SP": 0.0005, "INK": 0.001}


# ----------------------------------------------------------------------------- utils
def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def walk_files(root="/kaggle/input"):
    """Recursive file list. followlinks=True is REQUIRED: recent Kaggle images mount
    datasets as symlinks under /kaggle/input, which os.walk skips by default."""
    out, seen = [], set()
    for r, ds, fs in os.walk(root, followlinks=True):
        real = os.path.realpath(r)
        if real in seen:                      # guard against symlink cycles
            ds[:] = []; continue
        seen.add(real)
        for f in fs:
            out.append(os.path.join(r, f).replace(os.sep, '/'))
    return out


def load_rgb_resized(path, size=IMG):
    from PIL import Image
    im = Image.open(path)
    try:
        im.draft("RGB", (size * 2, size * 2))      # fast JPEG decoding at reduced scale
    except Exception:
        pass
    a = np.asarray(im.convert("RGB"))
    return cv2.resize(a, (size, size), interpolation=cv2.INTER_AREA)


def load_mask_resized(path, size=IMG):
    from PIL import Image
    m = np.asarray(Image.open(path).convert("L"))
    m = cv2.resize(m, (size, size), interpolation=cv2.INTER_AREA)
    return (m >= 128).astype(np.uint8)


def phash64(img):
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    g = cv2.resize(g, (32, 32), interpolation=cv2.INTER_AREA)
    d = cv2.dct(g)[:8, :8].flatten()
    return d > np.median(d[1:])


def thumbs(img):
    """z-normalised 32x32 grey thumbnails of the full image and of the central 50% crop
    (used to VERIFY pHash duplicate candidates; the centre crop ignores vignettes)."""
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    out = []
    for a in (g, g[64:192, 64:192]):
        t = cv2.resize(a, (32, 32), interpolation=cv2.INTER_AREA).flatten()
        t = (t - t.mean()) / (t.std() + 1e-6) / np.sqrt(t.size)
        out.append(t.astype(np.float16))
    return out[0], out[1]


def hamming_pairs(bitsA, bitsB=None, thr=6, device="cuda", chunk=2048):
    """Return (i, j, ham) with ham <= thr. If bitsB is None: pairs inside A with i<j."""
    dev = device if torch.cuda.is_available() else "cpu"
    dt = torch.float16 if dev != "cpu" else torch.float32
    A = torch.from_numpy(bitsA.astype(np.float32) * 2 - 1).to(dev, dt)
    B = A if bitsB is None else torch.from_numpy(bitsB.astype(np.float32) * 2 - 1).to(dev, dt)
    out = []
    for s in range(0, len(A), chunk):
        ham = (64 - (A[s:s + chunk] @ B.T).float()) / 2
        ii, jj = torch.nonzero(ham <= thr + 1e-3, as_tuple=True)
        h = ham[ii, jj]
        ii = ii + s
        if bitsB is None:
            keep = ii < jj
            ii, jj, h = ii[keep], jj[keep], h[keep]
        out.append(torch.stack([ii.float(), jj.float(), h], 1).cpu())
    o = torch.cat(out).numpy() if out else np.zeros((0, 3))
    return o[:, 0].astype(int), o[:, 1].astype(int), o[:, 2].astype(int)


class UnionFind:
    def __init__(self, n): self.p = np.arange(n)
    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]; a = self.p[a]
        return a
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb: self.p[max(ra, rb)] = min(ra, rb)


# ----------------------------------------------------------------------------- masks
def postprocess_lesion(prob, thr=0.5, min_frac=0.005):
    m = (prob >= thr).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return np.zeros_like(m), False
    j = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    m = (lab == j).astype(np.uint8)
    inv = 1 - m                                         # fill holes
    n2, lab2, _, _ = cv2.connectedComponentsWithStats(inv, 4)
    border = np.unique(np.concatenate([lab2[0], lab2[-1], lab2[:, 0], lab2[:, -1]]))
    holes = ~np.isin(lab2, border) & (inv > 0)
    m[holes] = 1
    return m, bool(m.mean() >= min_frac)


def otsu_lesion(img):
    """Classical fallback lesion segmentation (only used if no GT masks are available)."""
    b = cv2.GaussianBlur(img[..., 2], (11, 11), 0)
    _, m = cv2.threshold(b, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return postprocess_lesion(m.astype(np.float32))


def detect_artifacts(img, L):
    """Rule-based artifact masks on a 256x256 RGB uint8 image.
    L: uint8 lesion mask. Artifact pixels are never inside the (dilated) lesion.
    Returns uint8 bitfield (DC|TS|SP|INK)."""
    H, W = L.shape
    area = H * W
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    Hh, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    Ld = cv2.dilate(L.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))) > 0

    # (1) dark corners / vignette: dark regions connected to the image border
    dark = cv2.morphologyEx((V < 50).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(dark, 8)
    border = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    # must be a large blob (area) and not a thin line touching the border (hair): mean thickness >= 8 px
    keep = [b for b in border if b != 0 and st[b, cv2.CC_STAT_AREA] >= 0.002 * area and
            st[b, cv2.CC_STAT_AREA] / max(np.hypot(st[b, cv2.CC_STAT_WIDTH], st[b, cv2.CC_STAT_HEIGHT]), 1) >= 8]
    dc = np.isin(lab, keep) & ~Ld

    # (2) thin dark structures (hair, ruler ticks/lines): black-hat + elongation filter
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    bh = cv2.morphologyEx(cv2.GaussianBlur(gray, (3, 3), 0), cv2.MORPH_BLACKHAT, k)
    ts = (bh >= 18).astype(np.uint8)
    ts[cv2.dilate(dc.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0] = 0
    ts[:4, :] = 0; ts[-4:, :] = 0; ts[:, :4] = 0; ts[:, -4:] = 0     # v2: no frame-edge strips
    n, lab, st, _ = cv2.connectedComponentsWithStats(ts, 8)
    a = st[:, cv2.CC_STAT_AREA].astype(np.float64)
    diag = np.hypot(st[:, cv2.CC_STAT_WIDTH], st[:, cv2.CC_STAT_HEIGHT])
    good = (a >= 15) & (diag >= 15) & (a / np.maximum(diag, 1) <= 6.0)
    good[0] = False
    ts = good[lab] & ~Ld

    # (3) specular highlights / gel bubbles: small bright unsaturated blobs that are much brighter
    #     than their surround (v2: white top-hat keeps only bright peaks smaller than 11 px -> large
    #     whitish skin areas are rejected; bubble rims and glints are kept)
    tophat = cv2.morphologyEx(V, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    sp = ((V >= 210) & (S <= 50) & (tophat >= 25)).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(sp, 8)
    a = st[:, cv2.CC_STAT_AREA]
    good = (a >= 3) & (a <= 0.01 * area)
    good[0] = False
    sp = (cv2.dilate(good[lab].astype(np.uint8), np.ones((3, 3), np.uint8)) > 0) & ~Ld

    # (4) ink markings / coloured patches: strongly saturated blue-cyan-violet regions outside the lesion
    #     (v2: hue 90-150 also covers blue/cyan stickers; S>=90 rejects bluish-tinted skin; pink skin
    #      (H>150) excluded; a 12-px band next to the vignette is excluded (chromatic fringe))
    ink = cv2.morphologyEx(((Hh >= 90) & (Hh <= 150) & (S >= 90) & (V >= 40) & (V <= 235)).astype(np.uint8),
                           cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(ink, 8)
    good = st[:, cv2.CC_STAT_AREA] >= 40
    good[0] = False
    ink = good[lab] & ~Ld & ~(cv2.dilate(dc.astype(np.uint8), np.ones((25, 25), np.uint8)) > 0)

    return (dc * BIT_DC | ts * BIT_TS | sp * BIT_SP | ink * BIT_INK).astype(np.uint8)


def artifact_flags(mask_bits):
    """image-level presence flags from a (256,256) bitfield."""
    return {k: float(((mask_bits & b) > 0).mean()) >= ART_MIN_FRAC[k] for k, b in ART_BITS.items()}


# ----------------------------------------------------------------------------- GPU augmentation
def gpu_augment(img_u8, msk_u8, mode, out=OUT, gen=None):
    """img_u8 (B,256,256,3) uint8, msk_u8 (B,256,256) uint8 (on device).
    mode: 'eval' | 'std' | 'robust'. Image and masks receive the SAME geometric transform.
    Returns normalized image (B,3,out,out), lesion mask, artifact mask (B,1,out,out) in [0,1]."""
    dev = img_u8.device
    B = img_u8.shape[0]
    x = img_u8.permute(0, 3, 1, 2).float() / 255.0
    m = msk_u8.long()
    mL = ((m & BIT_L) > 0).float()[:, None]
    mA = ((m & BIT_ART) > 0).float()[:, None]
    R = lambda *s: torch.rand(*s, device=dev, generator=gen)
    theta = torch.zeros(B, 2, 3, device=dev)
    if mode == "eval":
        theta[:, 0, 0] = 1; theta[:, 1, 1] = 1
    elif mode == "std":       # Resize(256) + RandomCrop(224) + HFlip
        s = out / IMG
        fx = torch.where(R(B) < 0.5, -1.0, 1.0)
        theta[:, 0, 0] = s * fx; theta[:, 1, 1] = s
        theta[:, 0, 2] = (R(B) * 2 - 1) * (1 - s); theta[:, 1, 2] = (R(B) * 2 - 1) * (1 - s)
    elif mode == "robust":    # RandomResizedCrop(scale .4-1, ratio 3/4-4/3) + H/V flips (+ colour jitter)
        area = 0.4 + 0.6 * R(B)
        logr = math.log(3 / 4) + (math.log(4 / 3) - math.log(3 / 4)) * R(B)
        r = torch.exp(logr)
        sx = torch.sqrt(area * r).clamp(max=1.0); sy = torch.sqrt(area / r).clamp(max=1.0)
        fx = torch.where(R(B) < 0.5, -1.0, 1.0); fy = torch.where(R(B) < 0.5, -1.0, 1.0)
        theta[:, 0, 0] = sx * fx; theta[:, 1, 1] = sy * fy
        theta[:, 0, 2] = (R(B) * 2 - 1) * (1 - sx); theta[:, 1, 2] = (R(B) * 2 - 1) * (1 - sy)
    else:
        raise ValueError(mode)
    grid = F.affine_grid(theta, (B, 3, out, out), align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)
    mm = F.grid_sample(torch.cat([mL, mA], 1), grid, mode="bilinear", padding_mode="border", align_corners=False)
    mL, mA = mm[:, :1], mm[:, 1:]
    if mode == "robust":      # ColorJitter(brightness=.2, contrast=.2, saturation=.2)
        b = (0.8 + 0.4 * R(B)).view(B, 1, 1, 1)
        x = (x * b).clamp(0, 1)
        c = (0.8 + 0.4 * R(B)).view(B, 1, 1, 1)
        g = (0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]).mean((1, 2)).view(B, 1, 1, 1)
        x = ((x - g) * c + g).clamp(0, 1)
        s = (0.8 + 0.4 * R(B)).view(B, 1, 1, 1)
        gy = (0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2])[:, None]
        x = (gy + (x - gy) * s).clamp(0, 1)
    mean = torch.tensor(MEAN, device=dev).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=dev).view(1, 3, 1, 1)
    return (x - mean) / std, mL, mA


# ----------------------------------------------------------------------------- models
def densenet_features():
    from torchvision.models import densenet121, DenseNet121_Weights
    return densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1).features


class DenseCNN(nn.Module):
    """Black-box reference: DenseNet-121 + GAP + linear."""
    kind = "cnn"
    def __init__(self, n_classes=2, pretrained=True):
        super().__init__()
        self.features = densenet_features() if pretrained else _tiny_features()
        self.head = nn.Linear(1024, n_classes)
    def forward(self, x):
        f = F.relu(self.features(x))
        return {"logits": self.head(f.mean((2, 3))), "fmap": f}


class ProtoPNet(nn.Module):
    """ProtoPNet (Chen et al., NeurIPS 2019) with a DenseNet-121 backbone."""
    kind = "proto"
    def __init__(self, n_classes=2, ppc=10, D=128, pretrained=True):
        super().__init__()
        self.features = densenet_features() if pretrained else _tiny_features()
        self.add_on = nn.Sequential(nn.Conv2d(1024, D, 1), nn.ReLU(), nn.Conv2d(D, D, 1), nn.Sigmoid())
        P = n_classes * ppc
        self.prototypes = nn.Parameter(torch.rand(P, D, 1, 1))
        self.register_buffer("proto_class", torch.arange(P) // ppc)
        ident = F.one_hot(torch.arange(P) // ppc, n_classes).float().T          # (C,P)
        self.register_buffer("class_identity", ident)
        self.last = nn.Linear(P, n_classes, bias=False)
        with torch.no_grad():
            self.last.weight.copy_(1.0 * ident - 0.5 * (1 - ident))
        self.n_classes, self.P, self.D = n_classes, P, D

    def latent(self, x):
        f = F.relu(self.features(x))
        with torch.autocast("cuda", enabled=False):
            return self.add_on(f.float())

    def distances(self, z):
        x2 = (z ** 2).sum(1, keepdim=True)
        p2 = (self.prototypes ** 2).sum((1, 2, 3)).view(1, -1, 1, 1)
        xp = F.conv2d(z, self.prototypes)
        return F.relu(x2 + p2 - 2 * xp)

    def forward(self, x):
        z = self.latent(x)
        with torch.autocast("cuda", enabled=False):
            d = self.distances(z)
            sim = torch.log((d + 1) / (d + 1e-4))
            act = sim.amax((2, 3))
            logits = self.last(act)
        return {"logits": logits, "sim": sim, "dist": d, "min_d": d.amin((2, 3)), "act": act, "z": z}


def _tiny_features():  # CPU unit tests only
    return nn.Sequential(nn.Conv2d(3, 1024, 32, stride=32))


def build_model(cfg, pretrained=True):
    if cfg["model"] == "cnn":
        return DenseCNN(pretrained=pretrained)
    return ProtoPNet(ppc=cfg.get("ppc", 10), D=cfg.get("D", 128), pretrained=pretrained)


# ----------------------------------------------------------------------------- losses
def proto_base_losses(out, y, model, class_w):
    ce = F.cross_entropy(out["logits"], y, weight=class_w)
    md = out["min_d"]
    own = model.proto_class[None, :] == y[:, None]
    big = torch.finfo(md.dtype).max / 4
    clst = md.masked_fill(~own, big).amin(1).mean()
    sep = md.masked_fill(own, big).amin(1).mean()
    return ce, clst, sep


def artifact_lesion_losses(out, y, model, cL, cA, valid, tau=0.5):
    """Proposed losses.  sim: (B,P,h,w); cL/cA: lesion/artifact fraction per latent cell (B,1,h,w).
    L_art : activation-weighted spatial-attention mass that every prototype puts on artifact cells.
    L_les : -log attention mass that own-class prototypes put on lesion cells."""
    sim = out["sim"]
    B, P, h, w = sim.shape
    att = F.softmax(sim.flatten(2) / tau, dim=2)                         # (B,P,hw)
    wt = sim.flatten(2).amax(2).detach()
    wt = wt / wt.sum(1, keepdim=True).clamp_min(1e-8)
    a_art = (att * cA.flatten(2)).sum(2)                                  # (B,P)
    l_art_i = (wt * a_art).sum(1)
    own = (model.proto_class[None, :] == y[:, None]).float()
    w_own = wt * own
    w_own = w_own / w_own.sum(1, keepdim=True).clamp_min(1e-8)
    a_les = (att * cL.flatten(2)).sum(2)
    l_les_i = (w_own * -torch.log(a_les + 1e-4)).sum(1)
    has_les = valid & (cL.flatten(1).sum(1) >= 0.5)
    zero = sim.sum() * 0
    l_art = l_art_i[valid].mean() if valid.any() else zero
    l_les = l_les_i[has_les].mean() if has_les.any() else zero
    return l_art, l_les, a_art.detach(), a_les.detach()


def fine_annotation_loss(out, y, model, cL, valid):
    """IAIA-BL style fine-annotation loss (Barnett et al., 2021) adapted to lesion masks:
    own-class prototypes are penalised for activating outside the lesion, other-class
    prototypes for activating anywhere."""
    sim = out["sim"]
    own = model.proto_class[None, :] == y[:, None]
    t_own = (sim * (1 - cL)).flatten(2).norm(dim=2)
    t_oth = sim.flatten(2).norm(dim=2)
    per = (t_own * own).sum(1) + (t_oth * ~own).sum(1)
    return per[valid].mean() if valid.any() else sim.sum() * 0


def cell_fractions(mL, mA, hw):
    return F.adaptive_avg_pool2d(mL, hw), F.adaptive_avg_pool2d(mA, hw)


# ----------------------------------------------------------------------------- data
class ArrDS(torch.utils.data.Dataset):
    def __init__(self, img_path, msk_path, labels, valid, idx):
        self.img_path, self.msk_path = img_path, msk_path
        self.labels = np.asarray(labels).astype(np.int64)
        self.valid = np.asarray(valid).astype(bool)
        self.idx = np.asarray(idx)
        self.imgs = self.msks = None
    def __len__(self): return len(self.idx)
    def __getitem__(self, k):
        if self.imgs is None:
            self.imgs = np.load(self.img_path, mmap_mode="r")
            self.msks = np.load(self.msk_path, mmap_mode="r")
        i = int(self.idx[k])
        return (torch.from_numpy(np.ascontiguousarray(self.imgs[i])),
                torch.from_numpy(np.ascontiguousarray(self.msks[i])),
                int(self.labels[i]), bool(self.valid[i]), i)


def make_loader(ds, bs, shuffle, seed=0, workers=4, drop_last=False):
    g = torch.Generator(); g.manual_seed(seed)
    return torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=shuffle, generator=g, num_workers=workers,
                                       pin_memory=torch.cuda.is_available(), drop_last=drop_last,
                                       persistent_workers=workers > 0)


# ----------------------------------------------------------------------------- inference
@torch.no_grad()
def predict(model, loader, device, keep_maps=False):
    model.eval()
    probs, ys, ids, maps, acts = [], [], [], [], []
    for img, msk, y, v, i in loader:
        x, _, _ = gpu_augment(img.to(device, non_blocking=True), msk.to(device, non_blocking=True), "eval")
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            out = model(x.contiguous(memory_format=torch.channels_last))
        probs.append(F.softmax(out["logits"].float(), 1)[:, 1].cpu()); ys.append(y); ids.append(i)
        if keep_maps and "sim" in out:
            maps.append(out["sim"].half().cpu()); acts.append(out["act"].cpu())
    r = {"prob": torch.cat(probs).numpy(), "y": torch.cat(ys).numpy(), "idx": torch.cat(ids).numpy()}
    if keep_maps and maps:
        r["sim"] = torch.cat(maps).numpy(); r["act"] = torch.cat(acts).numpy()
    return r


def auc_safe(y, p):
    from sklearn.metrics import roc_auc_score
    try:
        return float(roc_auc_score(y, p))
    except Exception:
        return float("nan")


# ----------------------------------------------------------------------------- push
@torch.no_grad()
def push_prototypes(model, loader, device, constrained, les_min=0.5, art_max=0.05):
    """Project every prototype onto its nearest latent patch of a training image of its class.
    constrained=True: candidate patches must lie on the lesion (lesion fraction >= les_min)
    and be artifact-free (artifact fraction <= art_max)."""
    model.eval()
    P, D = model.P, model.D
    best_d = torch.full((P,), float("inf"), device=device)
    best_z = model.prototypes.data.view(P, D).clone()
    info = [dict(proto=j, cls=int(model.proto_class[j]), img=-1, h=-1, w=-1, dist=float("inf"),
                 les_frac=float("nan"), art_frac=float("nan")) for j in range(P)]
    for img, msk, y, v, ii in loader:
        img, msk, y, v = img.to(device), msk.to(device), y.to(device), v.to(device)
        x, mL, mA = gpu_augment(img, msk, "eval")
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            z = model.latent(x.contiguous(memory_format=torch.channels_last))
        d = model.distances(z.float())                                    # (B,P,h,w)
        B, _, h, w = d.shape
        cL, cA = cell_fractions(mL, mA, (h, w))
        ok = torch.ones(B, 1, h, w, dtype=torch.bool, device=device)
        if constrained:
            ok = (cL >= les_min) & (cA <= art_max) & v.view(B, 1, 1, 1)
        own = model.proto_class.view(1, P, 1, 1) == y.view(B, 1, 1, 1)
        dd = d.masked_fill(~(own & ok), float("inf")).permute(1, 0, 2, 3).reshape(P, -1)
        val, arg = dd.min(1)
        upd = val < best_d
        for j in torch.nonzero(upd).flatten().tolist():
            a = int(arg[j]); b, rem = divmod(a, h * w); hh, ww = divmod(rem, w)
            best_d[j] = val[j]; best_z[j] = z[b, :, hh, ww].float()
            info[j].update(img=int(ii[b]), h=hh, w=ww, dist=float(val[j]),
                           les_frac=float(cL[b, 0, hh, ww]), art_frac=float(cA[b, 0, hh, ww]))
    model.prototypes.data.copy_(best_z.view(P, D, 1, 1))
    return pd.DataFrame(info)


# ----------------------------------------------------------------------------- training
DEFAULT = dict(model="proto", aug="robust", lam_clst=0.8, lam_sep=0.08, lam_l1=1e-4,
               lam_art=0.0, lam_les=0.0, lam_fa=0.0, tau=0.5, push="standard",
               ep_warm=5, ep_joint=15, ep_ll=5, bs=64, lr_feat=1e-4, lr_addon=3e-3, lr_proto=3e-3,
               lr_head=1e-3, lr_ll=1e-3, wd=1e-3, seed=0)

CONFIGS = {
    "CNN":         dict(model="cnn"),
    "PPN-std":     dict(aug="std"),
    "PPN":         dict(),
    "PPN-FA":      dict(lam_fa=1e-3),
    "Ours":        dict(lam_art=1.0, lam_les=1.0, push="constrained"),
    "Ours-noLes":  dict(lam_art=1.0, lam_les=0.0, push="constrained"),
    "Ours-noArt":  dict(lam_art=0.0, lam_les=1.0, push="constrained"),
    "Ours-stdPush": dict(lam_art=1.0, lam_les=1.0, push="standard"),
}


def full_cfg(name, seed):
    c = dict(DEFAULT); c.update(CONFIGS[name]); c["name"] = name; c["seed"] = seed
    return c


def _set_trainable(model, stage):
    for p in model.parameters(): p.requires_grad = False
    if model.kind == "cnn":
        for p in model.head.parameters(): p.requires_grad = True
        if stage == "joint":
            for p in model.features.parameters(): p.requires_grad = True
        return
    if stage in ("warm", "joint"):
        for p in model.add_on.parameters(): p.requires_grad = True
        model.prototypes.requires_grad = True
        if stage == "joint":
            for p in model.features.parameters(): p.requires_grad = True
    elif stage == "last":
        for p in model.last.parameters(): p.requires_grad = True


def train_run(cfg, prep, outdir, device, smoke=False, workers=4, log=print):
    """prep: dict(img_path, msk_path, meta DataFrame with split,label,lesion_valid)."""
    os.makedirs(outdir, exist_ok=True)
    set_seed(cfg["seed"])
    torch.backends.cudnn.benchmark = True
    meta = prep["meta"]
    tr = np.where(meta.split.values == "train")[0]
    va = np.where(meta.split.values == "val")[0]
    if smoke:
        rs = np.random.RandomState(0)
        tr, va = rs.choice(tr, min(512, len(tr)), replace=False), rs.choice(va, min(256, len(va)), replace=False)
        cfg = dict(cfg, ep_warm=1, ep_joint=1, ep_ll=1)
    labels, valid = meta.label.values, meta.lesion_valid.values
    mk = lambda idx: ArrDS(prep["img_path"], prep["msk_path"], labels, valid, idx)
    tr_loader = make_loader(mk(tr), cfg["bs"], True, cfg["seed"], workers, drop_last=True)
    tr_eval_loader = make_loader(mk(tr), 128, False, 0, workers)
    va_loader = make_loader(mk(va), 128, False, 0, workers)
    n1 = labels[tr].sum(); n0 = len(tr) - n1
    class_w = torch.tensor([len(tr) / (2 * n0), len(tr) / (2 * n1)], dtype=torch.float32, device=device)

    model = build_model(cfg, pretrained=prep.get("pretrained", True)).to(device).to(memory_format=torch.channels_last)
    is_proto = model.kind == "proto"
    if is_proto:
        opt_warm = torch.optim.Adam([{"params": model.add_on.parameters(), "lr": cfg["lr_addon"], "weight_decay": cfg["wd"]},
                                     {"params": [model.prototypes], "lr": cfg["lr_proto"]}])
        opt_joint = torch.optim.Adam([{"params": model.features.parameters(), "lr": cfg["lr_feat"], "weight_decay": cfg["wd"]},
                                      {"params": model.add_on.parameters(), "lr": cfg["lr_addon"], "weight_decay": cfg["wd"]},
                                      {"params": [model.prototypes], "lr": cfg["lr_proto"]}])
    else:
        opt_warm = torch.optim.Adam(model.head.parameters(), lr=cfg["lr_head"])
        opt_joint = torch.optim.Adam([{"params": model.features.parameters(), "lr": cfg["lr_feat"], "weight_decay": cfg["wd"]},
                                      {"params": model.head.parameters(), "lr": cfg["lr_head"]}])
    sched = torch.optim.lr_scheduler.StepLR(opt_joint, step_size=5, gamma=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    hist, best = [], dict(auc=-1, state=None, epoch=-1)
    E = cfg["ep_warm"] + cfg["ep_joint"]
    t_start = time.time()

    def one_epoch(opt, stage):
        # last-layer stage: everything below the last layer is frozen INCLUDING BatchNorm statistics,
        # so the latent space (and hence the pushed prototypes = real training patches) stays exact.
        model.train(stage != "last")
        agg = {}
        for img, msk, y, v, ii in tr_loader:
            img, msk = img.to(device, non_blocking=True), msk.to(device, non_blocking=True)
            y, v = y.to(device), v.to(device)
            x, mL, mA = gpu_augment(img, msk, cfg["aug"])
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                out = model(x.contiguous(memory_format=torch.channels_last))
            parts = {}
            if is_proto:
                ce, clst, sep = proto_base_losses(out, y, model, class_w)
                parts.update(ce=ce, clst=clst, sep=sep)
                if stage == "last":
                    l1 = (model.last.weight * (1 - model.class_identity)).norm(p=1)
                    loss = ce + cfg["lam_l1"] * l1
                else:
                    loss = ce + cfg["lam_clst"] * clst - cfg["lam_sep"] * sep
                    if cfg["lam_art"] > 0 or cfg["lam_les"] > 0:
                        cL, cA = cell_fractions(mL, mA, out["sim"].shape[2:])
                        la, ll, _, _ = artifact_lesion_losses(out, y, model, cL, cA, v, cfg["tau"])
                        loss = loss + cfg["lam_art"] * la + cfg["lam_les"] * ll
                        parts.update(art=la, les=ll)
                    if cfg["lam_fa"] > 0:
                        cL, _ = cell_fractions(mL, mA, out["sim"].shape[2:])
                        fa = fine_annotation_loss(out, y, model, cL, v)
                        loss = loss + cfg["lam_fa"] * fa
                        parts.update(fa=fa)
            else:
                loss = F.cross_entropy(out["logits"].float(), y, weight=class_w)
                parts.update(ce=loss)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            for k, t in parts.items():
                agg[k] = agg.get(k, 0.0) + float(t.detach())
            agg["n"] = agg.get("n", 0) + 1
        n = agg.pop("n", 1)
        return {k: round(t / n, 4) for k, t in agg.items()}

    for ep in range(1, E + 1):
        stage = "warm" if ep <= cfg["ep_warm"] else "joint"
        _set_trainable(model, stage)
        t0 = time.time()
        tl = one_epoch(opt_warm if stage == "warm" else opt_joint, stage)
        if stage == "joint": sched.step()
        pv = predict(model, va_loader, device)
        auc = auc_safe(pv["y"], pv["prob"])
        rec = dict(epoch=ep, stage=stage, val_auc=round(auc, 4), sec=round(time.time() - t0, 1), **tl)
        hist.append(rec); log(f"[{cfg['name']} s{cfg['seed']}] {rec}")
        if not is_proto and auc > best["auc"]:
            best = dict(auc=auc, state=copy.deepcopy({k: t.cpu() for k, t in model.state_dict().items()}), epoch=ep)

    push_info = None
    if is_proto:
        push_info = push_prototypes(model, tr_eval_loader, device, constrained=cfg["push"] == "constrained")
        pv = predict(model, va_loader, device)
        auc = auc_safe(pv["y"], pv["prob"])
        hist.append(dict(epoch=E, stage="after_push", val_auc=round(auc, 4)))
        log(f"[{cfg['name']} s{cfg['seed']}] after push val_auc={auc:.4f} | protos on artifact cells(>0.5): "
            f"{int((push_info.art_frac > 0.5).sum())}/{len(push_info)} | on lesion cells(>=0.5): "
            f"{int((push_info.les_frac >= 0.5).sum())}/{len(push_info)}")
        best = dict(auc=auc, state=copy.deepcopy({k: t.cpu() for k, t in model.state_dict().items()}), epoch=E)
        _set_trainable(model, "last")
        opt_ll = torch.optim.Adam(model.last.parameters(), lr=cfg["lr_ll"])
        for k in range(1, cfg["ep_ll"] + 1):
            t0 = time.time()
            tl = one_epoch(opt_ll, "last")
            pv = predict(model, va_loader, device)
            auc = auc_safe(pv["y"], pv["prob"])
            rec = dict(epoch=E + k, stage="last", val_auc=round(auc, 4), sec=round(time.time() - t0, 1), **tl)
            hist.append(rec); log(f"[{cfg['name']} s{cfg['seed']}] {rec}")
            if auc > best["auc"]:
                best = dict(auc=auc, state=copy.deepcopy({k2: t.cpu() for k2, t in model.state_dict().items()}), epoch=E + k)
        push_info.to_csv(os.path.join(outdir, "push_info.csv"), index=False)

    if best["state"] is None:   # e.g. undefined val AUC: never save an empty checkpoint
        best = dict(auc=float("nan"), state={k: t.cpu() for k, t in model.state_dict().items()}, epoch=len(hist))
    torch.save(best["state"], os.path.join(outdir, "model.pt"))
    json.dump(dict(cfg=cfg, hist=hist, best_epoch=best["epoch"], best_val_auc=best["auc"],
                   minutes=round((time.time() - t_start) / 60, 1)), open(os.path.join(outdir, "log.json"), "w"), indent=1)
    log(f"[{cfg['name']} s{cfg['seed']}] DONE best_val_auc={best['auc']:.4f} at epoch {best['epoch']} "
        f"({(time.time() - t_start) / 60:.1f} min)")
    return best["auc"]


def find_prep_dirs(files=None):
    """Locate NB00 output (dir with meta.csv + images.npy) and NB00b output (dir with meta.csv + masks.npy +
    dup_candidates.csv), independent of how Kaggle mounts notebook outputs. Prints a diagnosis if missing."""
    files = files if files is not None else walk_files("/kaggle/input") + walk_files("/kaggle/working")
    by_dir = {}
    for f in files:
        by_dir.setdefault(os.path.dirname(f), set()).add(os.path.basename(f))
    p1 = sorted(d for d, fs in by_dir.items() if {"meta.csv", "images.npy"} <= fs)
    p2 = sorted(d for d, fs in by_dir.items() if {"meta.csv", "masks.npy", "dup_candidates.csv"} <= fs)
    if not p1:
        print("!!! NB00 output NOT found under /kaggle/input. What IS mounted (first 3 levels):")
        for d in sorted({"/".join(f.split("/")[:6]) for f in files})[:40]:
            print("   ", d)
        print("   any meta.csv:", [f for f in files if f.endswith("meta.csv")][:10])
    return (p1[0] if p1 else None), (p2[0] if p2 else None)
