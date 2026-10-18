"""Frozen-method stress test on Piti Yindee's original human-drawn Wuffle sheet.

Run from the repository: .venv/bin/python experiments/artist_sheets.py
Only this evaluation harness changes; reconstruction and paint fitting stay frozen.
"""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

from desina.core import Camera, Model, View, bake_paint, reconstruct, render, save_image
from desina.demo import metrics, panel
from desina.paint import PaintOptions, fit_paint
from desina.paint_benchmark import detail_error


ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'examples/artist-sheets'
SIZE = 192
RESOLUTION = 80
SEEDS = [41, 73]
OPTIONS = PaintOptions()
CENTERS = [139, 303, 440, 582, 750, 919, 1056, 1197]
TRAIN = [0, 2, 4, 6]
TEST = [1, 3, 5, 7]
# Background seeds preserve enclosed arm/leg gaps, rather than filling them as skin.
# Coordinates are in the original sheet. These are silhouette hints, never matches.
OUTFITS = {
    'signature': {'top': 432, 'center_y': 561, 'gaps': [
        [[94, 573], [182, 574]], [[276, 566], [307, 638]], [],
        [[555, 568], [584, 638]], [[699, 566], [791, 574]],
        [[943, 568], [915, 638]], [], [[1223, 566], [1192, 638]]]},
    'pajamas': {'top': 701, 'center_y': 824, 'gaps': [
        [[97, 840], [182, 840]], [[283, 831], [307, 906]], [],
        [[586, 908]], [[703, 831], [791, 840]], [[915, 907]], [], [[1194, 906]]]},
}
FROZEN = 'e6e2689b74f34f5d14a1a84fa0899d6dc51b117b'
FROZEN_FILES = ['core.py', 'paint.py', 'paint_joint.py']


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2) + '\n')


def prepare(directory):
    original = directory / 'original.jpg'
    image = np.asarray(Image.open(original).convert('RGB')) / 255.
    manifest = {
        'frozen_method_commit': FROZEN,
        'source_sha256': {name: hashlib.sha256((ROOT/'src/desina'/name).read_bytes()).hexdigest()
                          for name in FROZEN_FILES},
        'art_sha256': hashlib.sha256(original.read_bytes()).hexdigest(),
        'author': 'Piti Yindee', 'art_license': 'CC0-1.0',
        'download_repository': 'hjyoon02/ModelSheetTo3D',
        'download_commit': '569363513b6f847b2363ea71b6bcdbff2bcb3c3e',
        'download_path': 'data/ModelSheets/001/original.jpg',
        'size': SIZE, 'geometry_resolution': RESOLUTION,
        'options': asdict(OPTIONS), 'sampler_seeds': SEEDS,
        'train_indices': TRAIN, 'held_out_indices': TEST,
        'azimuth_degrees': [-45*i % 360 for i in range(8)],
        'camera_elevation': 0, 'camera_scale': 2.4,
        'column_centers': CENTERS, 'outfits': OUTFITS,
        'crop_width': 178, 'crop_height': 243, 'source_canvas_span': 282,
        'mask_ink_threshold': .78,
        'normalization': 'Only per-column translation and one common scale; no local alignment or stretching.',
        'protocol': 'Final framing fixed after a pilot exposed world-cube clipping at the feet. All final runs regenerated. Two outfits, both sampler seeds, all four held-out views reported; no fitter or hyperparameter changes.',
        'pose_control': 'One training view: smooth image deformation, RGB AND mask; maximum displacement 4 source pixels. Not an articulated 3D pose or new human drawing.',
        'contour_control': 'Uniform cream surface plus black two-pixel image silhouette outlines, from the human masks; diagnostic, not additional artist artwork.',
        'limitations': 'One artist, one character, two outfits; approximate cameras and no 3D ground truth. Held-out agreement is image agreement, not shape accuracy.',
    }
    write_json(directory/'manifest.json', manifest)
    result = {}
    for outfit, spec in OUTFITS.items():
        case = directory/outfit
        case.mkdir(exist_ok=True)
        views = []
        entries = []
        for index, cx in enumerate(CENTERS):
            top = spec['top']
            rgb = image[top:top+243, cx-89:cx+89].copy()
            ink = rgb.min(-1) < .78
            filled = ndi.binary_fill_holes(ink)
            labels, count = ndi.label(filled)
            sizes = np.bincount(labels.ravel()); sizes[0] = 0
            mask = labels == sizes.argmax()
            gaps, _ = ndi.label(~ink)
            for x, y in spec['gaps'][index]:
                label = gaps[y-top, x-(cx-89)]
                if label == 0:
                    raise ValueError(f'Background seed falls on ink: {outfit}/{index}: {(x, y)}')
                gap = gaps == label
                if np.any(gap[[0, -1]]) or np.any(gap[:, [0, -1]]):
                    raise ValueError(f'Background seed is not an enclosed gap: {outfit}/{index}')
                mask[gap] = False
            # Map a common 282-source-pixel square into the renderer's square canvas.
            # Every outfit uses the same height and scale; only row/column translation varies.
            yy, xx = np.mgrid[:SIZE, :SIZE]
            u = (xx+.5)*282/SIZE - 141 + 89 - .5
            v = (yy+.5)*282/SIZE - 141 + spec['center_y']-top - .5
            output_mask = ndi.map_coordinates(mask.astype(float), [v, u], order=0, mode='constant') > .5
            output_rgb = np.stack([ndi.map_coordinates(rgb[..., c], [v, u], order=1,
                                    mode='constant', cval=.96) for c in range(3)], -1)
            output_rgb[~output_mask] = .96
            name = f'view-{index}.png'
            save_image(case/name, output_rgb, output_mask)
            camera = Camera(-45*index % 360, 0, 2.4)
            foreground_coordinates = (np.argwhere(output_mask)+.5-SIZE/2)*camera.scale/SIZE
            if np.max(np.abs(foreground_coordinates)) >= 1:
                raise ValueError(f'Input extends beyond the world cube: {outfit}/{index}')
            views.append(View(camera, output_rgb, output_mask, name))
            entries.append({'image': name, 'camera': vars(camera)})
        write_json(case/'project.json', {'views': [entries[i] for i in TRAIN]})
        write_json(case/'held-out.json', {'views': [entries[i] for i in TEST],
            'purpose': 'Evaluation only: not passed to reconstruction or paint fitting.'})
        panel(case/'inputs.png', [(f'Train {i}: {views[i].camera.azimuth} deg', views[i].rgb) for i in TRAIN])
        panel(case/'held-out-drawings.png', [(f'Unseen drawing {i}', views[i].rgb) for i in TEST])
        result[outfit] = views
    return result


def score(pred, pmask, view):
    row = metrics(pred, pmask, view.rgb, view.mask)
    row['gradient_mae'] = detail_error(pred, view.rgb, pmask & view.mask)
    row['input_foreground_recall'] = float((pmask & view.mask).sum()/view.mask.sum())
    return row


def evaluated(model, views, indices, case, prefix):
    rows, renders = [], []
    for i in indices:
        rgb, mask, _ = render(model, views[i].camera, SIZE, False)
        rows.append({'view': i, **score(rgb, mask, views[i])})
        renders.append(rgb)
        save_image(case/f'{prefix}-view-{i}.png', rgb)
    return rows, renders


def pose_control(views):
    # A local 4-source-pixel displacement moves image contours and paint together.
    # Apply it to one front-view forearm region, without using any model correspondence.
    view = views[0]
    yy, xx = np.mgrid[:SIZE, :SIZE]
    du = 4*SIZE/282 * np.exp(-(((xx/SIZE-.72)/.13)**2 + ((yy/SIZE-.55)/.17)**2)/2)
    coordinates = [yy, xx-du]
    rgb = np.stack([ndi.map_coordinates(view.rgb[..., c], coordinates, order=1, mode='nearest')
                    for c in range(3)], -1)
    mask = ndi.map_coordinates(view.mask.astype(float), coordinates, order=0, mode='nearest') > .5
    rgb[~mask] = .96
    return View(view.camera, rgb, mask, 'pose-control-front.png')


def run_case(case, views):
    train = [views[i] for i in TRAIN]
    baseline = reconstruct(train, RESOLUTION)
    bake_paint(baseline, train, OPTIONS.texture_resolution)
    baseline.save(case/'baseline.npz')
    training_rows, _ = evaluated(baseline, views, TRAIN, case, 'baseline')
    baseline_rows, base_renders = evaluated(baseline, views, TEST, case, 'baseline')
    record = {'baseline': {'train': training_rows, 'held_out': baseline_rows}, 'fitted': []}
    preview_model = None
    for seed in SEEDS:
        fitted = Model(baseline.sdf.copy(), baseline.color.copy(), dict(baseline.metadata))
        fit = fit_paint(fitted, train, replace(OPTIONS, seed=seed),
                        progress=lambda line: print(f'{case.name}/{seed}: {line}', flush=True))
        np.testing.assert_array_equal(fitted.sdf, baseline.sdf)
        rows, predictions = evaluated(fitted, views, TEST, case, f'fitted-{seed}')
        record['fitted'].append({'seed': seed, 'held_out': rows, 'fit_seconds': fit['seconds'],
            'initial_energy': fit['initial_energy'], 'export_energy': fit.get('export_energy'),
            'sdf_unchanged': True})
        if seed == SEEDS[0]:
            # Fixed seed, chosen before any results; not the best held-out result.
            fitted.save(case/'fitted.npz')
            preview_model = fitted
        for j, i in enumerate(TEST):
            panel(case/f'comparison-{seed}-view-{i}.png', [
                ('Held-out human drawing', views[i].rgb), ('Projection baseline', base_renders[j]),
                (f'Shared texture / seed {seed}', predictions[j])],
                'Piti Yindee / CC0. Approximate camera; image agreement only, no 3D ground truth.')
    # Geometry-only render separates missing shape from misplaced painted lines.
    clay = Model(baseline.sdf.copy(), np.full_like(baseline.color, .7), dict(baseline.metadata))
    panel(case/'geometry.png', [(f'Clay / unseen {i}', render(clay, views[i].camera, SIZE, True)[0])
                              for i in TEST], 'Same frozen hull as the painted renders; no input paint.')
    # Visualize damage from incompatible input silhouettes before texture fitting.
    for i in TRAIN:
        rgb, pmask, _ = render(baseline, views[i].camera, SIZE, False)
        diagnostic = np.full_like(rgb, .96)
        diagnostic[views[i].mask & pmask] = [.25, .55, .40]
        diagnostic[views[i].mask & ~pmask] = [.90, .15, .23]
        diagnostic[~views[i].mask & pmask] = [.20, .40, .90]
        panel(case/f'training-silhouette-{i}.png', [('Input drawing', views[i].rgb),
            ('Training-view reconstruction', rgb), ('Green overlap / red lost', diagnostic)])
    # Leave-one-view-out hulls locate which constraint deletes which pixels.
    ablation = []
    for leave_out in TRAIN:
        model = reconstruct([views[i] for i in TRAIN if i != leave_out], RESOLUTION)
        rgb, mask, _ = render(model, views[0].camera, SIZE, False)
        ablation.append({'omitted_view': leave_out, 'front': score(rgb, mask, views[0]),
            'volume_fraction_vs_full': float((model.sdf <= 0).sum()/(baseline.sdf <= 0).sum())})
        save_image(case/f'omitted-{leave_out}-front.png', rgb)
    record['leave_one_out'] = ablation
    # Small local pose-like image change. Geometry is reconstructed again; painting is not refit.
    moved = pose_control(views)
    save_image(case/'pose-control-front.png', moved.rgb, moved.mask)
    perturbed_train = [moved] + [views[i] for i in TRAIN[1:]]
    perturbed = reconstruct(perturbed_train, RESOLUTION)
    save_image(case/'pose-control-shape.png', render(perturbed, Camera(315, 0, 2.4), SIZE, False)[0])
    a, b = baseline.sdf <= 0, perturbed.sdf <= 0
    record['pose_control'] = {'max_displacement_source_pixels': 4,
        'input_mask_changed_fraction': float(np.logical_xor(moved.mask, views[0].mask).sum()/views[0].mask.sum()),
        'removed_original_volume_fraction': float((a & ~b).sum()/a.sum()),
        'added_volume_fraction': float((~a & b).sum()/a.sum()),
        'symmetric_difference_fraction': float(np.logical_xor(a, b).sum()/a.sum()),
        'canonical_front': score(*render(perturbed, views[0].camera, SIZE, False)[:2], views[0])}
    # Isolate screen-space silhouette contours from surface paint, keeping human masks.
    cream = np.array([.85, .76, .59])
    contour_views = []
    for view in train:
        rgb = np.full_like(view.rgb, .96)
        rgb[view.mask] = cream
        rgb[view.mask & ~ndi.binary_erosion(view.mask, iterations=2)] = .03
        contour_views.append(View(view.camera, rgb, view.mask, view.name))
    contour = Model(baseline.sdf.copy(), baseline.color.copy(), dict(baseline.metadata))
    bake_paint(contour, contour_views, OPTIONS.texture_resolution)
    contour_rows = []
    for i in TEST:
        rgb, mask, _ = render(contour, views[i].camera, SIZE, False)
        interior = ndi.binary_erosion(mask, iterations=4)
        dark = rgb.mean(-1) < .35
        boundary = mask & ~ndi.binary_erosion(mask, iterations=2)
        contour_rows.append({'view': i, 'dark_interior_fraction': float((dark & interior).sum()/max(1, interior.sum())),
            'dark_boundary_recall': float((dark & boundary).sum()/max(1, boundary.sum()))})
        desired = np.full_like(rgb, .96); desired[mask] = cream; desired[boundary] = .03
        panel(case/f'contour-control-{i}.png', [('Baked image contours', rgb),
            ('Boundary-only diagnostic target', desired)],
            'Same reconstructed mask in both: isolates contour rendering from geometry errors.')
    record['contour_control'] = contour_rows
    frames = []
    for angle in range(0, 360, 15):
        rgb, _, _ = render(preview_model, Camera(angle, 0, 2.4), 144, False)
        frames.append(Image.fromarray(np.uint8(rgb*255)))
    frames[0].save(case/'turntable.gif', save_all=True, append_images=frames[1:], duration=150, loop=0)
    write_json(case/'metrics.json', record)
    return record


def summarize(results):
    summary = {}
    for outfit, data in results.items():
        rows = data['baseline']['held_out']
        fit_rows = [row for run in data['fitted'] for row in run['held_out']]
        keys = ['silhouette_iou', 'foreground_psnr_db', 'overlap_color_mae', 'gradient_mae']
        summary[outfit] = {'baseline_held_out': {key: float(np.mean([r[key] for r in rows])) for key in keys},
            'fitted_held_out': {key: float(np.mean([r[key] for r in fit_rows])) for key in keys},
            'baseline_training_foreground_recall': float(np.mean([r['input_foreground_recall'] for r in data['baseline']['train']])),
            'baseline_training_iou': float(np.mean([r['silhouette_iou'] for r in data['baseline']['train']])),
            'pose_control': data['pose_control'],
            'contour_control': {key: float(np.mean([r[key] for r in data['contour_control']])) for key in
                                ['dark_interior_fraction', 'dark_boundary_recall']}}
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    views = prepare(DEST)
    if args.prepare_only:
        return
    results = {}
    for outfit, outfit_views in views.items():
        print(f'Running {outfit}', flush=True)
        results[outfit] = run_case(DEST/outfit, outfit_views)
    summary = summarize(results)
    write_json(DEST/'summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
