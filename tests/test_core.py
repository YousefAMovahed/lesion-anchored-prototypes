"""
CPU-only sanity tests for the core components (no GPU, no real data needed).

    python tests/test_core.py

Checks:
  1. artifact detectors: true positives fire, typical false positives do not, lesion is never marked;
  2. joint GPU-style augmentation keeps image and masks aligned and has the right shapes;
  3. a full (tiny) training run for every configuration finishes and saves a checkpoint;
  4. after the push + last-layer stage every prototype is exactly equal to its source latent patch;
  5. the lesion-constrained push only selects lesion cells without artifacts;
  6. the memory-free CAM equals autograd Grad-CAM up to the constant 1/(h*w);
  7. counterfactual interventions never change a lesion pixel.
"""
import os, sys, tempfile, shutil, warnings
import numpy as np, pandas as pd, cv2, torch

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import artproto as ap   # noqa: E402
import evalkit as ek    # noqa: E402

DEV = torch.device("cpu")


def synthetic_image(rs):
    im = np.full((256, 256, 3), (200, 160, 150), np.uint8)
    im = (im.astype(int) + rs.randint(-6, 7, im.shape)).clip(0, 255).astype(np.uint8)
    cx, cy, r = 128 + rs.randint(-15, 15), 128 + rs.randint(-15, 15), rs.randint(35, 55)
    cv2.circle(im, (cx, cy), r, (90, 50, 40), -1)
    cv2.line(im, (0, rs.randint(10, 60)), (255, rs.randint(10, 60)), (25, 20, 20), 2)       # hair
    cv2.rectangle(im, (0, 200), (40, 255), (5, 5, 5), -1)                                    # dark corner
    L = np.zeros((256, 256), np.uint8); cv2.circle(L, (cx, cy), r, 1, -1)
    return im, L


def test_detectors():
    base = np.full((256, 256, 3), (205, 170, 165), np.uint8)
    L = np.zeros((256, 256), np.uint8); cv2.circle(L, (128, 128), 30, 1, -1)
    f = lambda x, b: int(((ap.detect_artifacts(x, L) & b) > 0).sum())
    hair = base.copy(); cv2.line(hair, (0, 20), (240, 60), (20, 20, 20), 2)
    dc = base.copy(); cv2.rectangle(dc, (226, 0), (255, 30), (5, 5, 5), -1)
    spec = base.copy(); cv2.circle(spec, (60, 60), 3, (255, 255, 255), -1)
    bubble = base.copy(); cv2.circle(bubble, (200, 70), 15, (250, 250, 250), 2)
    ink = base.copy(); cv2.line(ink, (20, 200), (120, 240), (120, 40, 160), 4)
    white_patch = base.copy(); cv2.circle(white_patch, (60, 60), 40, (240, 238, 236), -1)
    pink = np.full((256, 256, 3), (215, 120, 170), np.uint8)
    assert f(hair, ap.BIT_TS) > 0 and f(hair, ap.BIT_DC) == 0, "hair must be TS, not DC"
    assert f(dc, ap.BIT_DC) > 0, "dark corner not detected"
    assert f(spec, ap.BIT_SP) > 0 and f(bubble, ap.BIT_SP) > 0, "specular/bubble not detected"
    assert f(ink, ap.BIT_INK) > 0, "ink not detected"
    assert f(white_patch, ap.BIT_SP) == 0, "large whitish skin must not be SP"
    assert f(pink, ap.BIT_INK) == 0, "pink skin must not be INK"
    for img in (hair, dc, spec, bubble, ink):
        assert ((ap.detect_artifacts(img, L) > 0) & (L > 0)).sum() == 0, "artifact mask overlaps lesion"
    print("[ok] artifact detectors")


def make_dataset(d, n=96):
    rs = np.random.RandomState(0)
    imgs = np.lib.format.open_memmap(f"{d}/images.npy", mode="w+", dtype=np.uint8, shape=(n, 256, 256, 3))
    msks = np.lib.format.open_memmap(f"{d}/masks.npy", mode="w+", dtype=np.uint8, shape=(n, 256, 256))
    for i in range(n):
        im, L = synthetic_image(rs)
        imgs[i] = im; msks[i] = ap.detect_artifacts(im, L) | (L * ap.BIT_L)
    imgs.flush(); msks.flush()
    meta = pd.DataFrame(dict(split=["train"] * 64 + ["val"] * 32, label=(np.arange(n) % 3 == 0).astype(int),
                             lesion_valid=True))
    return dict(img_path=f"{d}/images.npy", msk_path=f"{d}/masks.npy", meta=meta, pretrained=False)


def test_augmentation(prep):
    imgs = np.load(prep["img_path"]); msks = np.load(prep["msk_path"])
    for mode in ("eval", "std", "robust"):
        x, mL, mA = ap.gpu_augment(torch.from_numpy(imgs[:4]), torch.from_numpy(msks[:4]), mode)
        assert x.shape == (4, 3, 224, 224) and mL.shape == (4, 1, 224, 224) and mA.shape == mL.shape
        assert 0 <= float(mL.min()) and float(mL.max()) <= 1
    print("[ok] joint augmentation")


def test_training_push_and_cam(prep, d):
    for name in ap.CONFIGS:
        cfg = ap.full_cfg(name, 0); cfg.update(ep_warm=1, ep_joint=1, ep_ll=1, bs=16)
        ap.train_run(cfg, prep, f"{d}/runs/{name}_s0", DEV, workers=0, log=lambda s: None)
        assert os.path.exists(f"{d}/runs/{name}_s0/model.pt")
    print("[ok] all", len(ap.CONFIGS), "configurations train and save a checkpoint")

    imgs = np.load(prep["img_path"]); msks = np.load(prep["msk_path"])
    for name in ("Ours", "PPN"):
        cfg = ap.full_cfg(name, 0)
        m = ap.build_model(cfg, pretrained=False); m.load_state_dict(torch.load(f"{d}/runs/{name}_s0/model.pt")); m.eval()
        pi = pd.read_csv(f"{d}/runs/{name}_s0/push_info.csv")
        err = 0.0
        for _, r in pi.iterrows():
            i, h, w, p = int(r.img), int(r.h), int(r.w), int(r.proto)
            x, _, _ = ap.gpu_augment(torch.from_numpy(imgs[[i]]), torch.from_numpy(msks[[i]]), "eval")
            with torch.no_grad():
                z = m.latent(x)
            err = max(err, float((z[0, :, h, w] - m.prototypes[p, :, 0, 0]).abs().max()))
        assert err < 1e-5, f"prototype differs from its source patch ({err})"
        if name == "Ours":
            assert (pi.les_frac >= 0.5).all() and (pi.art_frac <= 0.05).all(), "constrained push violated"
    print("[ok] prototypes equal their source patches; constrained push only uses clean lesion cells")

    m = ap.DenseCNN(pretrained=False).eval()
    cam = ek.gradcam(m, imgs, msks, np.arange(4), DEV)
    x, _, _ = ap.gpu_augment(torch.from_numpy(imgs[:4]), torch.from_numpy(msks[:4]), "eval")
    f = torch.relu(m.features(x)); f.retain_grad(); m.head(f.mean((2, 3)))[:, 1].sum().backward()
    ref = torch.relu((f.grad.mean((2, 3), keepdim=True) * f).sum(1)).detach().numpy()
    hw = ref.shape[1] * ref.shape[2]
    assert np.allclose(cam.astype(np.float32) / hw, ref, atol=1e-4 * max(1.0, float(ref.max()))), "CAM != Grad-CAM"
    print("[ok] memory-free CAM equals Grad-CAM up to 1/(h*w)")


def test_counterfactuals(prep):
    IM = np.load(prep["img_path"], mmap_mode="r"); MS = np.load(prep["msk_path"], mmap_mode="r")
    for kind in ek.INTERVENTIONS:
        CF, CH, ok = ek.make_counterfactuals(IM, MS, np.arange(20), kind)   # asserts lesion invariance inside
        for k in np.where(ok)[0]:
            L = (MS[k] & ap.BIT_L) > 0
            assert np.array_equal(CF[k][L], np.array(IM[k])[L])
    print("[ok] counterfactual interventions keep every lesion pixel unchanged")


if __name__ == "__main__":
    d = tempfile.mkdtemp()
    try:
        test_detectors()
        prep = make_dataset(d)
        test_augmentation(prep)
        test_training_push_and_cam(prep, d)
        test_counterfactuals(prep)
        print("\nALL TESTS PASSED")
    finally:
        shutil.rmtree(d, ignore_errors=True)
