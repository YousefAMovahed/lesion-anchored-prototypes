# Fig. 1 (vector PDF): (a) the six automatic-mask examples of the former Fig. 1 in one row; (b) method overview: a real ISIC test
# image with its lesion mask, detected dark corners and the 7x7 cells admissible for the constrained projection, and the pipeline.
# (b) uses row 1 of fig/explanations_3rows.png (lesion contour red, artifacts cyan); cell fractions are computed at the
# 224 x 224 input resolution as in the method (average pooling to 7 x 7).
import numpy as np, cv2, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch
from PIL import Image

import os
FIG = os.path.dirname(os.path.abspath(__file__))  # run from paper/fig: python make_overview.py
OUT = FIG + '/overview.pdf'

# ---- (a) six tiles of the former Fig. 1 (2 x 3 grid, tiles at rows 2-241 / 246-485, cols 2-241 / 246-485 / 490-729)
am = np.array(Image.open(FIG + '/artifact_masks.png').convert('RGB'))
R_, C_ = [(2, 242), (246, 486)], [(2, 242), (246, 486), (490, 730)]
tiles = [am[r0:r1, c0:c1] for (r0, r1) in R_ for (c0, c1) in C_]

# ---- (b) example image and masks
src = np.array(Image.open(FIG + '/explanations_3rows.png').convert('RGB'))
prof = (src[26:228, 0:40].astype(int).sum(2) < 740).sum(0)
x0 = int(np.argmax(prof > 150))
img = src[26:228, x0:x0 + 202].copy()
r, g, b = [img[..., k].astype(int) for k in range(3)]
red = (r - g > 60) & (r - b > 60)                              # anti-aliased red lesion contour
ring = cv2.dilate(red.astype(np.uint8), np.ones((3, 3), np.uint8))
ff = ring.copy(); fm = np.zeros((ring.shape[0] + 2, ring.shape[1] + 2), np.uint8)
cv2.floodFill(ff, fm, (101, 5), 1)
les = ((ff == 0) | (ring == 1)).astype(np.uint8)
n, lab, st, _ = cv2.connectedComponentsWithStats(les)
les = (lab == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)   # largest component, as in the mask post-processing
cyan = (g > 150) & (b > 150) & (r < 120)
dark = (r + g + b) < 150
cand = ((dark | cyan) & (les == 0)).astype(np.uint8)
n, lab = cv2.connectedComponents(cand)
edge = set(np.unique(np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]])) - {0}
dc = np.isin(lab, list(edge)).astype(np.uint8)                  # dark corners: dark region touching the border
ov = cv2.dilate((red | cyan).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
clean = img.copy(); clean[ov] = cv2.medianBlur(img, 9)[ov]      # image without the drawn contours

def cells(m):
    m224 = cv2.resize(m.astype(np.float32), (224, 224), interpolation=cv2.INTER_LINEAR)
    return m224.reshape(7, 32, 7, 32).mean((1, 3))
ML, MA = cells(les), cells(dc)
adm = (ML >= 0.5) & (MA <= 0.05)
print('cells with lesion >= 0.5:', int((ML >= 0.5).sum()), '| admissible:', int(adm.sum()), '| cells with artifact > 0.05:', int((MA > 0.05).sum()))

plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'], 'mathtext.fontset': 'stix',
                     'font.size': 7, 'axes.linewidth': 0.4})
W = 3.45
lx, gap = 0.13, 0.011                                           # left margin for the panel labels, gap between tiles
tw = (W - lx - 5 * gap) / 6                                     # tile width (in)
s = 1.06                                                        # size of the example image in (b)
yb0 = 0.02; ya0 = yb0 + s + 0.10                                # bottom of (b) and of (a)
H = ya0 + tw + 0.02
fig = plt.figure(figsize=(W, H))
F = lambda x, y: (x / W, y / H)

def show(x, y, w, a):
    ax = fig.add_axes([x / W, y / H, w / W, w / H]); ax.imshow(a); ax.set_xticks([]); ax.set_yticks([])
    for s_ in ax.spines.values(): s_.set_linewidth(0.3)
    return ax
xs = lx
for k, t in enumerate(tiles):
    show(xs + k * (tw + gap), ya0, tw, t)
fig.text(*F(0.0, ya0 + tw / 2), '(a)', fontsize=7, rotation=90, va='center', ha='left')

ax = show(lx, yb0, s, cv2.resize(clean, (224, 224), interpolation=cv2.INTER_AREA))
ax.contour(cv2.resize(les.astype(np.float32), (224, 224)), levels=[0.5], colors=['#e00000'], linewidths=0.8)
dcm = np.ma.masked_where(cv2.resize(dc.astype(np.float32), (224, 224)) < 0.5, np.ones((224, 224)))
ax.imshow(dcm, cmap=matplotlib.colors.ListedColormap(['#00c8c8']), alpha=0.6, vmin=0, vmax=1)
for k in range(1, 7):
    ax.axhline(32 * k - 0.5, color='w', lw=0.3, alpha=0.9); ax.axvline(32 * k - 0.5, color='w', lw=0.3, alpha=0.9)
for i in range(7):
    for j in range(7):
        if adm[i, j]:
            ax.add_patch(Rectangle((32 * j - 0.5, 32 * i - 0.5), 32, 32, fill=False, ec='#22cc22', lw=1.0))
fig.text(*F(0.0, yb0 + s / 2), '(b)', fontsize=7, rotation=90, va='center', ha='left')

x1, x2 = lx + s + 0.07, W - 0.005
boxes = [('Masks (training images only): lesion fraction $M^L$\nand detected-artifact fraction $M^A$ of each cell', 0.245),
         ('Training: $\\mathcal{L}_{\\mathrm{les}}$ moves the attention $\\alpha_j$ of every\nprototype onto the lesion (Section III-B)', 0.245),
         ('Constrained push: $p_j\\leftarrow$ closest admissible cell\n($M^L\\geq0.5$, $M^A\\leq0.05$; green) of class $c(j)$', 0.245),
         ('Last-layer refit; inference needs no masks', 0.14)]
ah = (s - sum(h for _, h in boxes)) / (len(boxes) - 1)          # vertical space for the arrows
y = yb0 + s
for k, (t, h) in enumerate(boxes):
    fig.patches.append(FancyBboxPatch(F(x1, y - h), (x2 - x1) / W, h / H, boxstyle='round,pad=0.003,rounding_size=0.01',
                                      transform=fig.transFigure, fc='#f2f2f2', ec='#555555', lw=0.4))
    fig.text(*F(x1 + 0.045, y - h / 2), t, fontsize=6.5, va='center', ha='left', linespacing=1.2)
    if k < len(boxes) - 1:
        xm = (x1 + x2) / 2
        fig.patches.append(FancyArrowPatch(F(xm, y - h - 0.004), F(xm, y - h - ah + 0.006), transform=fig.transFigure,
                                           arrowstyle='-|>', mutation_scale=5, lw=0.5, color='#333333'))
    y -= h + ah
fig.savefig(OUT, dpi=600)
print('saved; height in:', round(H, 3))
