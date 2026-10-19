# Lesion-Anchored Prototypes

**Reducing Artifact Shortcuts in Interpretable Melanoma Classification**

Yousef Azizi Movahed, Fatemeh Ziaeetabar
Department of Computer Science, School of Mathematics, Statistics, and Computer Science, College of Science, University of Tehran

This repository contains the complete code, the executed notebooks and the logs behind the paper (CSICC 2026 submission). All results in the paper can be traced to the files in `results/`.

<p align="center">
  <img src="assets/explanations_3rows.png" width="720"><br>
  <em>ISIC 2019 test images with artifacts. The black-box CNN and an annotation-supervised ProtoPNet (PPN-FA) respond to dark corners and borders; the most active prototype of our model stays on the lesion.</em>
</p>

## Summary

Melanoma classifiers trained on dermoscopic archives pick up acquisition artifacts such as dark vignetting corners, rulers and ink. In a prototypical part network (ProtoPNet) the explanation itself can then point at the artifact. We

1. **audit the bias** on ISIC 2019 with automatically detected artifact masks, a leakage-free split and source-stratified statistics (dark corners: Mantel–Haenszel OR 1.56 [1.43, 1.69]; bright structures: 1.75 [1.59, 1.92]);
2. **anchor prototypes to the lesion** with two components: a lesion-attraction loss on the spatial attention of each prototype and a projection ("push") restricted to artifact-free lesion patches, both using masks generated without any manual annotation (an additional artifact-suppression loss turned out to be redundant and is reported as a negative result);
3. **test decisions and explanations with counterfactuals**: rulers, hair, dark corners and ink are inserted (and real hair/ruler marks removed) while every lesion pixel stays bit-identical, with an inpainting-only null control;
4. **audit the masks and the split**: the conclusions are re-checked on test images with expert masks only, under ±5 px mask erosion/dilation and per acquisition source, and a leakage audit confirms that no group, lesion or near-duplicate pair is shared between train, validation and test (`results/mask_reliability/`).

All eight configurations use the same DenseNet-121 backbone, data split and training budget, and are trained with three seeds.

## Main results

ISIC 2019 test set (n = 2,320), mean ± std over three seeds; external AUC on unseen datasets.

| Model | AUC | BAcc | Seed-ensemble AUC [95% CI] | Derm7pt-derm | Derm7pt-clin | PAD-UFES-20 |
|---|---|---|---|---|---|---|
| CNN (black box) | .902 ± .006 | .814 ± .011 | .912 [.884, .935] | .799 ± .023 | .704 ± .022 | .721 ± .022 |
| ProtoPNet, std. aug. | .835 ± .013 | .752 ± .014 | .861 [.827, .888] | .677 ± .027 | .624 ± .023 | .587 ± .019 |
| ProtoPNet | .825 ± .040 | .758 ± .010 | .866 [.830, .894] | .688 ± .072 | .618 ± .038 | .665 ± .016 |
| ProtoPNet + IAIA-BL loss | .844 ± .008 | .744 ± .021 | .866 [.830, .895] | .696 ± .024 | .640 ± .013 | .630 ± .059 |
| **Ours** | **.865 ± .006** | **.780 ± .014** | **.884 [.850, .910]** | **.749 ± .009** | **.657 ± .019** | .647 ± .108 |

Where the models look and how they react to inserted artifacts (ISIC test):

| Model | Hot region on artifacts ↓ | Artifact enrichment ↓ | Prototypes on lesion (of 20) | Flip rate, +dark corners ↓ | Flip rate, +ruler ↓ | Map stability ρ, +dark corners ↑ |
|---|---|---|---|---|---|---|
| CNN | 23.1% | 1.52 | – | 6.1% | 5.6% | – |
| ProtoPNet | 6.6% | 0.46 | 14.3 | 10.9% | 7.5% | 0.68 |
| ProtoPNet + IAIA-BL | 10.2% | 0.68 | 10.0 | 10.6% | 6.5% | 0.71 |
| **Ours** | **4.8%** | **0.37** | **20.0** | **5.2%** | **3.6%** | **0.92** |

Ablations: removing lesion attraction lowers the test AUC to .830 (DeLong p = 8e-8) and doubles artifact use; removing the artifact-suppression term changes no metric significantly (negative result); the standard projection keeps the AUC but leaves only 7.7 of 20 prototypes on the lesion. Full tables: `paper/main.pdf` and `results/executed_notebooks/03_evaluate_executed.ipynb`.

<p align="center">
  <img src="assets/counterfactual_example_0.png" width="560"><br>
  <em>Counterfactual example with unchanged lesion pixels: map of the prototype most active on the original image, and the melanoma probability p.</em>
</p>

## Repository structure

```
src/
  artproto.py      data caching, lesion/artifact masks, joint GPU augmentation, ProtoPNet and CNN,
                   proposed losses, lesion-constrained push, training loop
  evalkit.py       metrics, DeLong test, bootstrap CIs, convex last-layer refit, attribution metrics,
                   CAM, counterfactual interventions, explanation-stability metrics
notebooks/
  01_prepare_data.ipynb        builds the leakage-free split, lesion masks (DeepLabV3), artifact masks,
                               bias statistics, QC figures and the external test sets
  02_train_all_models.ipynb    self-contained: prepares the data (same code as 01) and trains the
                               8 configurations x 3 seeds on two GPUs (resumable, deadline-aware)
  03_evaluate.ipynb            all results of the paper: classification, statistics, attribution,
                               counterfactuals, figures and LaTeX tables
results/
  executed_notebooks/          the notebooks exactly as they were run, with all outputs
  logs/                        full training and evaluation logs (24 runs, every epoch)
  mask_reliability/            data distribution, leakage audit, results by mask provenance,
                               under mask perturbation and per source (cell + full log)
tests/
  test_core.py                 CPU-only sanity tests of the core claims (see below)
paper/                         LaTeX source and PDF of the paper
```

## Reproducing the results

The experiments were run on Kaggle (2 x NVIDIA T4, PyTorch 2.11). Total compute: about 20 min data preparation, about 6.5 h training (24 runs of about 30 min on two GPUs) and about 40 min evaluation.

1. **Datasets** (add them as Kaggle inputs; the notebooks find them automatically):
   - ISIC 2019 training data: <https://www.kaggle.com/datasets/andrewmvd/isic-2019>
   - HAM10000 lesion masks: <https://www.kaggle.com/datasets/surajghuwalewala/ham1000-segmentation-and-classification>
   - Derm7pt: <https://www.kaggle.com/datasets/menakamohanakumar/derm7pt>
   - PAD-UFES-20: <https://www.kaggle.com/datasets/mahdavi1202/skin-cancer>
   - PH2 (segmentation validation only): <https://www.kaggle.com/datasets/athina123/ph2dataset>
2. Run `notebooks/02_train_all_models.ipynb` with *Save & Run All* (GPU T4 x2, Internet on). It prepares the data and trains all models; finished runs are reused if the notebook is restarted with its previous output attached.
3. Attach the output of step 2 to `notebooks/03_evaluate.ipynb` (set `TRAIN_NOTEBOOK_SLUG`) and run it. All tables and figures are written to `/kaggle/working/eval`.

`01_prepare_data.ipynb` is only needed to inspect the data preparation separately; notebook 02 contains the same code.

### Determinism

The data split is deterministic (seeded stratified group k-fold on fixed groups) and is checked by assertions in notebook 03 (train/val/test = 20,552 / 2,459 / 2,320). GPU training uses cuDNN autotuning, so re-runs reproduce the numbers up to small run-to-run variation; the three-seed means and the statistical tests are the quantities reported in the paper.

## Tests

```bash
pip install -r requirements.txt
python tests/test_core.py
```

The tests run on a CPU with synthetic images and check that

- the artifact detectors find hair, rulers, dark corners, specular highlights and ink, ignore typical false positives, and never mark lesion pixels;
- every configuration trains end to end and saves a checkpoint;
- after the projection and last-layer stage every prototype is **exactly** its source training patch, and the constrained push only uses artifact-free lesion cells;
- the memory-free CAM used for the CNN equals autograd Grad-CAM up to the factor 1/(h·w);
- the counterfactual interventions never change a single lesion pixel.

## Data and licenses

The code is released under the MIT license (`LICENSE`). The datasets are **not** redistributed here and keep their own licenses (ISIC 2019 and HAM10000: CC BY-NC 4.0; see the dataset pages for Derm7pt, PAD-UFES-20 and PH2). No patient data are included in this repository apart from the small example images in the figures, which come from the public ISIC 2019 archive.

## Citation

```bibtex
@inproceedings{azizimovahed2026lesionanchored,
  title     = {Lesion-Anchored Prototypes: Reducing Artifact Shortcuts in Interpretable Melanoma Classification},
  author    = {Azizi Movahed, Yousef and Ziaeetabar, Fatemeh},
  booktitle = {Proc. 30th International Computer Conference, Computer Society of Iran (CSICC)},
  year      = {2026}
}
```
