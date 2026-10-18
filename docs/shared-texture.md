# Shared-texture experiment

This iteration fixes the two reproduced implementation issues in the pairwise fitter and introduces a shared 3D texture likelihood. It remains an experiment using only artist-provided images/cameras or procedural stand-ins. No training datasets, neural networks, pretrained descriptors, named features, or landmark correspondences are involved.

## Run and inspect

```sh
uv run desina paint-demo --out examples/paint-shared
uv run desina studio examples/paint-shared/fitted.npz
uv run desina fit-paint model.npz drawings/project.json --out fitted.npz
uv run desina render fitted.npz --flat --out selected.png
uv run desina render fitted.npz --hypothesis 1 --flat --out alternative.png
uv run desina render fitted.npz --uncertainty --out uncertainty.png
uv run desina paint-benchmark --out examples/paint-validation
```

`fit-paint` and `paint-demo` now default to `--objective shared`. `--objective pairwise` retains the corrected previous objective for comparison. `paint-benchmark` uses its own fixed, smaller budget declared in `paint_benchmark.py`; its checked-in manifest records the exact settings and source hashes used for the first fresh evaluation.

`examples/paint/` and [the earlier report](paint-fitting.md) preserve the original v2 experiment, including its mixed metrics. `examples/paint-shared/` is the new development example. The fresh suite is stored separately in `examples/paint-validation/`.

## One texture in fitting and rendering

The old method optimized agreement between warped input images, then constructed the final texture using a separate source-selection heuristic. Its input-image objective did not measure the actual color field that the renderer would sample.

The new state contains both continuous image-warp controls **D** and a native 3D RGB volume **C**. At each fixed surface quadrature point x, the model compares:

- `C(x)`, sampled with `Model.sample` using precisely the same trilinear volume interpolation as rendering;
- each visible input image sampled at its projected location plus the displacement field `D_i`.

A visibility-weighted Gaussian-inlier/uniform-outlier likelihood scores those observations against that one shared color. Displacement, deformation smoothness, mask support, and a small edge-preserving color-difference prior complete the objective. There are no pairwise image-color agreement terms in the shared data likelihood.

The RGB volume written to the output is the selected joint state's actual volume. **There is no final re-fusion stage that can silently replace it with a different texture.** Regression tests reconstruct the objective from an exported model and verify that its score matches the selected-state score.

This is still a surface-quadrature approximation using fixed geometry and calibrated cameras. It is not a full likelihood for every observed image pixel, light source, garment layer, silhouette, and pose. In particular, paint can improve while silhouette or pose errors remain entirely unexplained.

## Alternating inference, with simple proposals

Warp proposals remain short, ordinary code: translation, affine tilt, and Gaussian bumps applied to low-resolution control fields. Conditional on a fixed texture, annealed Metropolis steps use the shared likelihood. The local acceptance delta is tested against recomputing the full objective.

Periodically, the texture gets two **data-driven candidate proposals**:

1. At each full-surface sample, choose an observed color mode minimizing the visibility-weighted shared observation loss.
2. Form an IRLS color estimate using the current texture's inlier responsibilities, to explain ordinary small RGB noise without giving large contradictions equal weight.

Each estimate is baked into a complete candidate RGB volume. The actual volume is then sampled at the fixed fitting quadrature, and its complete shared objective, including color regularization, is evaluated. A texture candidate is accepted only if it decreases that objective. Proposal heuristics do not determine the final output unless the joint score approves them.

Thus this is **alternating conditional Monte Carlo and monotone texture optimization**. The texture updates are not reversible posterior-sampling moves, and the resulting chain bank is not a converged joint posterior. This distinction matters: multiple retained interpretations improve inspection and uncertainty diagnostics, but do not establish posterior probabilities.

The knobs from the previous fitter remain. New controls are:

| Option | Default | Meaning |
| --- | ---: | --- |
| `--objective` | `shared` | Shared-volume likelihood or corrected pairwise comparison |
| `--texture-every` | 50 | Warp sweeps between canonical texture proposals |
| `--texture-prior` | 0.025 | Edge-preserving color regularization on fitting samples |

The blur scales now track input image resolution instead of being fixed pixel widths. Canonical RGB resolution remains independent of geometry resolution.

## Correctness fixes

**Best-state export:** previously, best states were recorded only every tenth sweep, and all improvements during the tail were discarded. Every accepted warp state is now tracked. Each chain's tail also contributes possible best states. A controlled regression drives all tail moves toward a better solution and checks that export beats every retained draw.

**Source-selection oscillation:** the corrected pairwise comparison uses one explicit undirected Potts graph. Graph coloring splits it into independent node groups; each group updates only on strict improvement. Since no adjacent nodes update simultaneously, the defined source energy cannot increase. The two-point oscillation reproduction now settles on a coherent assignment. This source-selection routine is retained for the pairwise comparison; shared output is the jointly selected RGB volume.

## Alternative interpretations and uncertainty

Each annealed chain preserves its own best controls and complete texture, including improvements found in its fixed-texture tail. Up to three candidate interpretations are serialized for inspection. Identity with a conditional texture fit is also considered as a fallback.

New optional `.npz` fields are:

- `paint_hypothesis_warps`: controls for saved candidates;
- `paint_hypothesis_colors`: their complete native RGB volumes;
- `paint_hypothesis_energies`: their scores under the same final-scale objective.

`--hypothesis N` renders one of these volumes. Rendering still never conditions on the input drawing or viewing direction. Existing `.npz` models without the extra arrays continue to load.

All chains retain short fixed-temperature, fixed-texture warp tails. Metadata associates each draw with its chain, and each candidate with its originating chain. The uncertainty visualization now includes candidate-texture disagreement as well as observation disagreement, sparse coverage, and warp spread. These components are heuristic and equally weighted across candidate modes; **the visualization is not calibrated**, and short tails can still reflect drift or miss alternatives. Subsequent manual edits do not refresh it.

## Fresh evaluation protocol

Development used the existing corruption seed 2026. Before generating fresh cases, the benchmark writes a manifest containing all settings, planned seeds, camera definitions, and SHA-256 hashes of relevant source files. The fitter settings remain unchanged throughout this suite; every listed case and sampler seed is reported.

The fresh protocol contains:

- Three new corruption seeds: 4101, 4102, 4103, at distortion amplitudes 0.02, 0.04, and 0.06 image widths.
- Two procedural paint designs: ragged tear-like marks and colored patchwork. Patchwork also receives a small unmodelled spatial tone variation.
- Two sampler seeds: 41 and 73.
- Four new withheld camera angles: 17/26, 73/-12, 152/31, and 237/-8 degrees.
- Shared geometry 56 cubed, texture 80 cubed, and input images 128 square for every method.
- Equal fitting budgets: 120 proposals per view/scale/chain, two chains, and 800 fixed random quadrature samples.

All 12 dataset/sampler combinations compare projection averaging, corrected pairwise fitting, and shared-texture fitting. The suite saves every run's four-camera metrics and a comparison at the same predeclared camera. Input PNGs and project files are included for all six datasets. Clean truth, corruption parameters, design IDs, and withheld images never enter `fit_paint`.

Metrics include overlap RGB MAE, foreground-union PSNR, and dense image-gradient MAE on eroded overlapping foreground. The latter adds a detail-sensitive check without identifying individual marks or matching edge features. The 48 angle evaluations per method are correlated measurements of six synthetic input sets, not 48 independent objects or artists.

## Results on the predeclared fresh suite

| Method | Overlap RGB MAE (lower) | Foreground PSNR (higher) | Gradient MAE (lower) |
| --- | ---: | ---: | ---: |
| baseline | 0.1361 | 12.14 dB | 0.0733 |
| pairwise | 0.1296 | 11.95 dB | 0.0750 |
| shared | 0.1269 | 12.13 dB | 0.0714 |

Aggregate shared-texture RGB error is **6.8% below projection averaging** and **2.1% below corrected pairwise fitting**. It improves over projection averaging in **11/12** runs and over corrected pairwise fitting in **6/12** runs. Relative to averaging, individual run gains range from a **1.5% regression** to a **17.7% improvement**. Foreground PSNR is essentially unchanged relative to averaging; gradient MAE improves about **2.6%**. These results support a modest aggregate benefit, with clear seed/style sensitivity.

| Dataset | Distortion | Baseline MAE | Pairwise MAE | Shared MAE |
| --- | ---: | ---: | ---: | ---: |
| rips-4101 | 0.020 | 0.0620 | 0.0562 | 0.0559 |
| rips-4102 | 0.040 | 0.0780 | 0.0726 | 0.0673 |
| rips-4103 | 0.060 | 0.0866 | 0.0772 | 0.0721 |
| patchwork-4101 | 0.020 | 0.1611 | 0.1639 | 0.1529 |
| patchwork-4102 | 0.040 | 0.2027 | 0.1883 | 0.1895 |
| patchwork-4103 | 0.060 | 0.2264 | 0.2192 | 0.2237 |

Dataset rows average both sampler seeds and all four cameras; individual runs remain in `runs.json`. The strongest-distortion patchwork case remains difficult. Its shared mean improves slightly over averaging but regresses against corrected pairwise fitting; one sampler seed also regresses against averaging. No parameters or case selection were changed after inspecting these results.

The full suite took about 181 seconds in this runtime while the development demo also ran concurrently. The original seed-2026 development example took about 41 seconds and reduced mean MAE from 0.0774 to 0.0640; this development improvement is deliberately separate from the fresh-suite evidence. In that example all 73 recorded texture proposals were non-increasing, and exported and selected joint energies both equal 0.44737905.

The legacy `fusion_only` key in the new development demo denotes shared-texture fitting with identity warps; it is not the old post-hoc source-fusion algorithm. The earlier v2 artifacts remain unchanged.

## Remaining limits

The geometry and visibility are fixed. There is no camera fitting, pose recovery, robust silhouette inference, illumination decomposition, true cloth-hole reconstruction, or automatic layer interpretation. Tear-like details are still paint. Distortion and local conflicts can be reconciled only within the bounded control-field family.

The color proposals use all available surface observations, but acceptance evaluates a fixed random quadrature approximation. Unmeasured regions can still be poorly constrained. The RGB color prior is spatial and has no semantic/material understanding. Repeated patterns and contradictory designs can produce plausible but incorrect interpretations.

The fresh suite is larger than the original single-corruption experiment, but uses the same simple trouser geometry and procedural drawing distortions. Actual artist drawings, varied geometry, pose differences, and multiple lighting/effect styles remain necessary before claiming practical robustness. No inference from these synthetic results alone establishes that readiness.

## Verification

Thirteen tests cover the original pipeline and the reviewed regressions. The shared-model checks include full/local objective agreement, native exported-volume score agreement, monotonically accepted texture updates, multiple saved chain interpretations, backward-compatible serialization, and view-independent rendering. The CLI fit/render/alternative/uncertainty round trip is also exercised. The desktop GUI is unchanged and remains untested interactively in the headless runtime.
