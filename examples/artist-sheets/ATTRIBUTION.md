# Artwork provenance

**Wuffle — Character Model Sheet (second version)**, by **Piti Yindee**, 9 June 2012.
The original human-drawn artwork is offered under [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/).
Credit is retained here even though CC0 does not require attribution. No endorsement is implied.

- [Primary artwork and license record on Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Piti_Yindee_-_Wuffle_-_Wuffle_-_Character_Model_Sheet_(second_version).jpg).
- [Original artist-posted source recorded by Commons](https://i1244.photobucket.com/albums/gg562/WuffleComic/CharactersSheets/CharactersSheetVer2_01_Wuffle.jpg).
- Downloaded byte-for-byte from [`data/ModelSheets/001/original.jpg`](https://github.com/hjyoon02/ModelSheetTo3D/blob/569363513b6f847b2363ea71b6bcdbff2bcb3c3e/data/ModelSheets/001/original.jpg), an independently hosted copy in Hyejeong Yoon and collaborators' public **ModelSheetTo3D** research repository. Commons downloads were blocked by this execution environment's host allowlist.

That repository explicitly distinguishes its original sheet from its AI-edited crops,
Gemini-generated views, generated line drawings, and TRELLIS model. **Only `original.jpg`
was used here.** None of its code, models, learned weights, processed images, segmentations,
camera assignments, or correspondences enter this experiment.

The source image is preserved as `original.jpg`; its SHA-256 is in `manifest.json`.
Our derived images use crops, an ink threshold, hole filling, a largest-component mask,
a few recorded background seeds to preserve enclosed silhouette gaps, a common scale,
and translation onto a square background. There is no local alignment, redrawing,
generative preprocessing, or color harmonization in the actual artist-data cases.
The pose and contour controls are explicitly synthetic transformations of these inputs.
Their definitions and every crop/mask parameter are in the manifest and evaluation script.

The original and derived artistic assets in this directory, including renderings, GIFs,
and saved volume artwork, are distributed under **CC0 1.0**. This notice does not change
the licensing of the reconstruction code. The evaluation is our responsibility;
the original artist and dataset hosts did not validate the reconstruction.
