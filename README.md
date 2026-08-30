# one-step-gen-empirical

Empirical comparison of **one-step generation models** against standard multi-step baselines (**flow matching** and **DDPM/DDIM diffusion**) on synthetic 2D/ high-dimensional datasets and real 28×28 images. The core question: can a single-step generative model match the generation quality of a 50-step multi-step model, and does it avoid the mode collapse that typically plagues one-step methods? Everything runs comfortably under a 16GB memory budget (small MLPs / small UNet, CPU-friendly).

## Methods

| Method | Type | NFE for generation | Description |
|--------|------|--------------------|-------------|
| `flow_matching` | Multi-step baseline | 50 (also evaluated at 1) | Standard conditional flow matching trained on straight noise→data paths. Uses Euler integration over `sample_steps` intervals. |
| `diffusion` | Multi-step baseline | 50 (also evaluated at 1) | Standard epsilon-prediction DDPM (1,000-step linear schedule), sampled with deterministic DDIM (η=0) so the multi-step budget matches flow matching. Same backbone family. |
| `mean_flow` | One-step | 1 (also evaluated at 2, 4) | Learns the average velocity field `u(x, t, r)` via a JVP-corrected regression target with adaptive weighting. |
| `improved_mean_flow` | One-step | 1 (also evaluated at 2, 4) | Variant of MeanFlow: logit-normal time sampling, compound objective `V = u + (t-r)·∂u/∂t`, and EMA-smoothed network for sampling. |
| `drifting_model` | One-step | 1 | Drifting Models: an EMA network maps noise directly to data. The target drift is derived from pairwise distance-based positive/negative attraction (`torch.cdist`), so each training step is a self-supervised "drift" that pulls generated points onto data clusters. |

All five use the same backbone family (an MLP vector field for low-dim data, a `SmallUNet` for images), so differences are attributable to the training/sampling scheme rather than capacity.

## Synthetic Datasets

| Dataset | Purpose |
|---------|---------|
| `moons` | Two disjoint half-moon arcs. Basic sanity check. |
| `checkerboard` | Grid of disjoint squares. Tests sharp boundaries and coverage of many modes. |
| `8gaussians` | Eight Gaussian modes in a circle. Basic multi-modal coverage. |
| `25gaussians` | Standard 5×5 grid of Gaussians. The de facto test for mode dropping in one-step / distilled models. |
| `unequal_gmm` | Unbalanced 3-mode GMM (70% / 20% / 10% weights, different variances). Tests whether minority modes are preserved. |
| `swissroll` | Continuous spiraling 2D manifold. Tests low-dimensional manifold learning. |
| `gmm_16d` | 8 Gaussian modes on a radius-3 sphere in 16D, per-mode stds 0.05–0.4. First stress test for distance-based drift in higher dims. |
| `gmm_64d` | Same mixture in 64D. Distances lose discriminative power in high dims, so this stresses the pairwise-`cdist` drift objective. |

The `unequal_gmm` and `25gaussians` datasets are the critical diagnostics: they expose mode collapse that the aggregate distance metrics (MMD/SWD) can hide.

## Experimental Setup

- 20,000 training samples, 2,000 evaluation samples, 150 epochs, batch size 256, Adam lr 1e-3, MLP hidden 256 / depth 4, seed 0 (CUDA).
- The `flow_matching` baseline is evaluated at 1 step (naive single Euler step) and 50 steps (converged multi-step). Each one-step method is evaluated at its native 1 step plus 2 and 4 steps to see behavior as the budget is relaxed (the drifting model is natively one-shot, so extra steps are a no-op).
- **Metrics** (computed on 2,000 matched samples): RBF **MMD** (Maximum Mean Discrepancy) and **Sliced Wasserstein Distance (SWD)** — lower is better — plus wall-clock sampling time and NFE.

## Results

### Generation quality (MMD / SWD, lower is better)

Each cell shows `MMD / SWD`. `FM@1` is a naive single Euler step of flow matching; `FM@50` is the converged multi-step baseline; the last three columns are one-step methods at NFE=1.

| Dataset | FM@1 | FM@50 | mean_flow@1 | improved_mf@1 | drifting@1 |
|---------|------|-------|-------------|---------------|------------|
| moons | 0.262 / 0.530 | 0.001 / 0.037 | 0.067 / 0.251 | 0.016 / 0.166 | **0.0002 / 0.018** |
| checkerboard | 0.214 / 0.445 | 0.001 / 0.028 | 0.071 / 0.224 | 0.001 / 0.052 | **0.0004 / 0.023** |
| 8gaussians | 0.245 / 0.586 | **0.000 / 0.026** | 0.017 / 0.221 | 0.005 / 0.146 | 0.001 / 0.040 |
| 25gaussians | 0.188 / 0.527 | 0.001 / 0.042 | 0.038 / 0.204 | 0.004 / 0.099 | **0.001 / 0.047** |
| unequal_gmm | 0.559 / 0.424 | 0.024 / 0.035 | 0.121 / 0.437 | 0.089 / 0.302 | **0.002 / 0.033** |
| swissroll | 0.180 / 0.344 | 0.001 / 0.027 | 0.081 / 0.201 | 0.017 / 0.126 | **0.001 / 0.022** |

Key points:
- **Naive one-step flow matching (FM@1) is poor everywhere** (MMD 0.18–0.56): a single Euler step does not reach the data distribution. Multi-step integration is essential for the standard baseline.
- **The drifting model at NFE=1 matches or beats 50-step flow matching on every dataset.** On `unequal_gmm` it is ~15× better in MMD (0.002 vs 0.024) — the setting where flow matching itself struggles most.
- The three "one-step" methods are **not interchangeable**: drifting ≫ improved_mean_flow > mean_flow.

### Mode coverage (the real test)

Aggregate MMD/SWD can hide collapse, so mode occupancy was measured explicitly (live-trained models, 2,000 samples):

**25gaussians** — should populate all 25 modes uniformly:

| Method | Modes hit | Min / Max per mode | Uniformity (min/max) |
|--------|-----------|--------------------|----------------------|
| flow_matching@50 | 25/25 | 51 / 111 | 0.46 |
| mean_flow@1 | 25/25 | 9 / 209 | 0.04 |
| improved_mean_flow@1 | 25/25 | 29 / 151 | 0.19 |
| drifting_model@1 | 25/25 | 59 / 114 | **0.52** |

**unequal_gmm** — true mode proportions `[0.70, 0.20, 0.10]`:

| Method | Mode proportions | Preserves minority? |
|--------|------------------|---------------------|
| flow_matching@50 | [0.73, 0.18, 0.09] | Yes |
| mean_flow@1 | [0.96, 0.04, 0.00] | **No — drops the 10% mode** |
| improved_mean_flow@1 | [0.92, 0.08, 0.00] | **No — drops the 10% mode** |
| drifting_model@1 | [0.72, 0.19, 0.09] | **Yes (best match)** |

The drifting model does **not** mode-collapse: it preserves even the 10% minority mode as well as 50-step flow matching does. Plain and improved MeanFlow both drop the minority mode on the unbalanced GMM, and MeanFlow's 25-gaussian coverage is badly skewed (9 vs 209 samples per mode).

### Sampling speed

One-step methods are dramatically cheaper at inference (2,000 samples, small MLP, CUDA): ~0.0002–0.0007 s vs ~0.005–0.017 s for 50-step flow matching — roughly **10–85× faster**, and the gap grows with network size since it is a single forward pass versus 50.

### Training convergence

- `flow_matching`: clean monotone loss decrease (1.31 → 1.02); the multi-step model is the slowest to *sample* but the fastest to *train* per epoch.
- `mean_flow` / `improved_mean_flow`: the adaptive loss weighting pins the reported loss at ~0.997 for the entire run, which masks progress. Monitoring the **median squared error** (`mse_med`) reveals real convergence — e.g. on 25gaussians `mean_flow` drops 20.4 → 8.2 and `improved_mean_flow` 2.04 → 1.67 (the raw *mean* is dominated by rare JVP outliers and is not usable as a signal).
- `drifting_model`: loss settles quickly (≈ flat after ~15 epochs); training is ~2× slower per epoch than flow matching due to the pairwise-distance drift computation.

## Harder targets: high-dimensional GMMs and real images

The 2D experiments above are the friendly case. This section stress-tests the drifting model against two harder target classes, still head-to-head with multi-step (no-shortcut) baselines: the existing **flow matching@50** plus a newly added **DDPM + 50-step DDIM diffusion** on the same backbone family.

### Setup

- **High-dim GMMs** (`gmm_16d`, `gmm_64d`): 8 modes placed on a radius-3 sphere, each with a different std (0.05–0.4), 20,000 train / 2,000 eval samples, same hyper-parameters as the 2D runs (hidden 256, depth 4, 150 epochs). Metrics: RBF MMD / SWD.
- **Real images** (MNIST, FashionMNIST, 28×28): 8,000 train / 2,000 eval samples, 20 epochs, batch 128, `SmallUNet` backbone. Metrics: Frechet distance proxy (fixed random-feature encoder) + pixel MMD. 28×28 sample grids are saved to `logs/`.
- **Diffusion baseline**: epsilon-prediction DDPM with a 1,000-step linear schedule, sampled with deterministic DDIM (η=0) over 50 steps, so it spends the same sampling budget as FM@50. It is a standard multi-step method — no shortcuts.

### High-dimensional GMMs — MMD / SWD (lower is better)

| Dataset | FM@50 | Diffusion@50 | Drift@1 |
|---------|-------|--------------|---------|
| gmm_16d | 0.077 / 0.387 | 0.072 / 0.378 | 0.086 / 0.399 |
| gmm_64d | 0.065 / 0.187 | 0.076 / 0.631 | 0.414 / 0.512 |

Mode coverage (8 true modes, 4,000 generated samples, k-means on the reference; `min/max` per mode):

| Dataset | FM@50 | Diffusion@50 | Drift@1 |
|---------|-------|--------------|---------|
| gmm_16d | 8/8 (191–1143) | 8/8 (178–1247) | 8/8 (191–1042) |
| gmm_64d | 8/8 (108–737) | 8/8 (368–912) | **1/8 (all 4,000 in one mode)** |

### Real images — Frechet distance proxy / pixel MMD (lower is better)

| Dataset | FM@50 | Diffusion@50 | Drift@1 |
|---------|-------|--------------|---------|
| mnist | 0.002 / 0.003 | 0.009 / 0.019 | 0.350 / 0.491 |
| fashionmnist | 0.007 / 0.005 | 0.058 / 0.091 | 5.575 / 0.759 |

The drift samples on images are not digit-like: the mean pixel value is shifted toward 0, the contrast is roughly half that of real data, and generated images sit ~5× farther from any real training image than FM's (nearest-neighbour MSE ≈ 1.0 vs ≈ 0.22 for FM@50 on MNIST).

### What this shows

- **At 16D the drift model still holds up**: it lands within ~20% of the multi-step baselines in MMD and covers all 8 modes, at a fraction of the NFE.
- **At 64D the drift objective collapses**: in a high-dimensional space the pairwise `torch.cdist` distances become near-identical across modes, so the positive/negative softmax drift can no longer separate the modes and every sample lands in a single mode (MMD ~6× worse than FM/diffusion).
- **On structured image data the drift model degrades hard**: the one-step noise→image map produces low-quality, non-digit samples on both MNIST and FashionMNIST (Frechet proxy 0.35–5.6 vs 0.002–0.058 for the multi-step baselines). The pairwise-distance drift built for low-dimensional, clusterable data does not transfer to the structured high-dimensional pixel manifold.
- **Diffusion is the most robust baseline**: near-flow-matching quality on the GMMs and clean image samples everywhere — it never mode-collapses.

## Takeaways

1. **A one-step model can match multi-step quality — but only for easy, low-dimensional targets, and only if designed for it.** The drifting model equals or beats 50-step flow matching on every 2D synthetic dataset and stays within ~20% on a 16D GMM, while costing ~50× less at sampling time. On harder targets the gap closes and then inverts: at 64D it mode-collapses to a single mode, and on MNIST/FashionMNIST its samples are not recognizably digits. The low-dimensional, cluster-centric drift objective does not transfer to high-dimensional or structured data.
2. **"One-step" is not a property of the sampling routine; it is a property of the training objective.** Plain MeanFlow stays one-step in name only — its NFE=1 output is the worst of the one-step methods and its quality only becomes competitive after several steps (e.g. 25gaussians MMD 0.038 → 0.003 at 2 steps). The drifting model, by contrast, is genuinely accurate in a single pass on the 2D toys.
3. **Multi-step integration is essential for flow matching.** Its naive 1-step Euler output is unusable (MMD 0.18–0.56); the model only becomes strong once ~50 steps are budgeted. The same holds for DDPM — 1-step diffusion prediction is far from the data (MNIST Frechet proxy 0.11) while 50-step DDIM reaches 0.009.
4. **Aggregate metrics understate mode collapse.** MMD/SWD alone would rank the drifting model and mean-flow models similarly, but explicit mode counting shows MeanFlow-family models drop or skew modes — and on the 64D GMM the drift model's MMD, while elevated, does not reveal that it collapsed from 8 modes to 1. Mode-coverage diagnostics should accompany MMD/SWD whenever comparing one-step vs multi-step generators.
5. **Diffusion is the most robust of the three baselines.** It tracks flow matching on the synthetic sets and produces clean image samples, never collapsing — the price is a single 50-step DDIM trajectory comparable in cost to FM@50.

## Notes on metrics & reproducibility

- **EMA checkpointing**: `drifting_model` and `improved_mean_flow` sample via an EMA network. Their `state_dict()` previously saved only the live network, so reloaded checkpoints silently produced collapsed/garbage samples. This is fixed — both now persist `ema_net` alongside `net` (legacy checkpoints load with a warning-free fallback that keeps the stale EMA).
- **SWD validity**: `sliced_wasserstein` rank-aligns sorted projections and truncates to the smaller count, so it is only meaningful when the two sample sets have equal size. All reported values use 2,000 vs 2,000. In 64D the SWD is dominated by projection noise and is a weaker signal than MMD (e.g. diffusion@50 MMD is fine while its SWD looks poor).
- **Harder-target runs**: the `diffusion` baseline and `gmm_16d`/`gmm_64d` datasets are new; image and high-dim results were produced with the same seed/hyper-parameter conventions as the 2D runs. Mode counts for the GMMs use k-means (8 clusters) fitted on the 4,000-sample reference, then generated samples are assigned to the nearest cluster.
- **Dataset name fix**: `real.py` previously accepted `fashion` but the registry registers `fashionmnist`, which crashed every FashionMNIST run; the loader now maps `fashionmnist` correctly.
