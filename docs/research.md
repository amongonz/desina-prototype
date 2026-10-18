# Research notes and experiment decisions

The objective is a first editable 3D representation whose unseen-angle render resembles the source, rather than a full reproduction of a published system. Research used primary author/project pages, paper abstracts, and official code READMEs. No implementation or pretrained assets from these projects were copied.

## Relevant methods

- [Lifting Freehand Concept Sketches into 3D](https://ns.inria.fr/d3/Lift3D/) (Gryaditskaya et al., 2020; [official implementation](https://github.com/ygryadit/LiftConceptSketches3D)) distinguishes true 3D intersections from accidental 2D occlusions in vector industrial design sketches, then infers stroke depth using scaffold geometry. This addresses the lifting ambiguity; a fully painted surface is an additional problem. Its construction-line cues could eventually replace some camera/depth hints.
- [Descriptive: Interactive 3D Shape Modeling from a Single Descriptive Sketch](https://doi.org/10.1016/j.cad.2020.102904) (Bobenrieth et al., 2020) partitions visible/hidden outline curves into closed elements and fits surfaces with optional local thickness/shape constraints. The useful design lesson is that small explicit artist constraints can resolve ambiguities. Desina currently uses camera/framing and masks, followed by local sphere-based shape correction; it does not implement Descriptive's optimizer.
- [NeuVAS](https://arxiv.org/abs/2506.13050) (Wang et al., 2025; [official implementation](https://github.com/GeometryArt/NeuVAS)) fits neural implicit surfaces to sparse **3D** shape constraints, including unstructured curve sketches, using curvature-based variation and feature-curve treatment. Its code expects a 3D point cloud; a 2D multi-view lifting stage would still be necessary. Per-object neural optimization is compatible with the no-pretraining restriction, but its higher-order surface losses are a more involved first CPU experiment.
- [Variational Neural Surfacing of 3D Sketches](https://zytadam.github.io/publications/vns/) (Zhang et al., 2025; [official implementation](https://github.com/zytadam/Variational-Neural-Surfacing-of-3D-Sketches)) uses an implicit neural representation and a variational objective inspired by principal-curvature flow lines in **3D** sketches. It motivates richer stroke-aligned surface regularization, but does not by itself solve camera inference or painted 2D sketch interpretation. The prototype's Gaussian smoothing is only a simple numerical regularizer, not an implementation of this method.

## Chosen baseline

A sampled signed-distance volume offers easy boolean sculpting, no mesh topology bookkeeping, deterministic CPU reconstruction, and persistent 3D paint. A calibrated silhouette visual hull is a transparent way to test the minimal success criterion. It makes its main ambiguity visible: silhouettes cannot recover concavities that never affect an outline. The smoothing step and paint fusion do not solve this geometric ambiguity.

The surface RGB field depends only on position. Its *construction* uses camera visibility, but its *rendering* never conditions on camera direction. This prevents a camera-dependent appearance trick from satisfying the unseen-view criterion. Unseen paint is extrapolated from the nearest observed shell; it is not hallucinated from a dataset.

## Monte Carlo options considered

Monte Carlo search is useful when plausible camera alignments, stroke correspondences, or depth hints produce multiple shape hypotheses. A practical next experiment would sample small azimuth/elevation/scale/translation perturbations, reconstruct each candidate, rank by silhouette reprojection plus complexity and artist constraints, and let the artist choose among several hypotheses. This searches uncertainty rather than averaging surfaces into one oversized volume. It was **not implemented** in the baseline: the synthetic inputs have known cameras, so a deterministic visual hull isolates whether geometry, visibility, paint, and editing work. Sampling alone cannot resolve hidden concavities without extra constraints.

A second option is per-object random-ray optimization of a grid or small randomly initialized SDF network against silhouettes, paint, sparse depth, and normal constraints. It would still use only artist-provided images. Before doing that, add camera fitting and a few 2D point correspondences; otherwise optimization can explain inaccurate sketches with implausible geometry.

## Reproducible results

`uv run desina demo --out examples/bird --resolution 80 --size 160`

Seven input views: azimuths 0, 60, 120, 180, 240, 300 at elevation 5 degrees, plus azimuth 35/elevation 65. Reconstruction takes about 1.4 seconds on this environment (not a general performance guarantee). Image size is 160 square and the implicit grid is 80 cubed. Visibility-weighted shell paint coverage is 89.0%; the remainder uses spatial nearest-color extension. This coverage refers to the two-voxel shell, not a precise area integral over the surface.

| Held-out azimuth / elevation | Silhouette IoU | Foreground PSNR | Overlap RGB MAE |
| --- | ---: | ---: | ---: |
| 30 / 22 | 0.9297 | 14.52 dB | 0.0424 |
| 145 / -12 | 0.9204 | 14.28 dB | 0.0421 |
| 275 / 35 | 0.8570 | 13.37 dB | 0.0498 |

Mean silhouette IoU: 0.9024. These are three views of one simple generated object, not evidence of general sketch reconstruction performance. The worst view exposes a larger hull error. Small eyes and boundaries visibly smear because the hull differs from the true ellipsoids and per-view color locations disagree on that hull.

## Limits and next experiments

- No perspective model, automatic camera pose/framing inference, or inconsistent-view alignment. Fixed cameras are explicit artist hints.
- Sparse views overestimate volume. Internal curves are currently paint, not depth or crease constraints. Occluded cavities need carving or new depth/normal hints.
- Closed outlines or explicit masks are required. Stroke classification, scaffold removal, and hidden-line inference are absent.
- Paint fusion averages conflicting drawings. It preserves neither independent artist layers nor editable source-to-surface correspondences; better UV-free paint layers and per-view override weights would help.
- No semantic prior, symmetry assumption, inferred anatomy, rig, relighting recovery, or pretrained generative model.
- Grid resolution limits small details. Fine features can disappear during carving/smoothing or fall between ray samples. Color-space fusion and the renderer are deliberately simple.
- Edits are spherical local operations, bounded by the world cube. They are not automatic, constraint-preserving propagation back into the source drawings.
- Desktop editing requires Tk and a display; automated validation covers the reconstruction/renderer/edit core, not real GUI interactions.

Next, test actual hand-painted orthographic sheets; ask for a few corresponding landmarks and one thickness hint; fit small camera offsets; add localized depth constraints and per-view paint priorities. Only then compare a per-object neural variational fit against this deterministic baseline on the same held-out cameras.
