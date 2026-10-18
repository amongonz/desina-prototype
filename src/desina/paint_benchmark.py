"""Predeclared fresh corruption sets; no clean evaluation images enter fitting."""
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy import ndimage as ndi

from .core import Camera, Model, bake_paint, load_project, reconstruct, render, save_image
from .demo import metrics, panel
from .paint import PaintOptions, fit_paint
from .paint_demo import inconsistent_paint, trouser_truth

SAMPLER_SEEDS = [41, 73]
CASES = [(4101, .02), (4102, .04), (4103, .06)]
DESIGNS = ['rips', 'patchwork']
OPTIONS = PaintOptions(steps=120, chains=2, samples=800, texture_resolution=80, posterior_draws=6)
HELD_OUT = [Camera(17, 26), Camera(73, -12), Camera(152, 31), Camera(237, -8)]


def detail_error(pred, truth, mask):
    interior = ndi.binary_erosion(mask, iterations=2)
    if not interior.any():
        return 0.
    # Dense image-gradient discrepancy; no named marks or matched edge features.
    return float(np.mean([np.abs(np.gradient(pred, axis=axis)-np.gradient(truth, axis=axis))[interior].mean()
                          for axis in [0, 1]]))


def run_benchmark(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).parent
    hashes = {name: hashlib.sha256((source/name).read_bytes()).hexdigest() for name in
              ['paint.py', 'paint_joint.py', 'core.py', 'paint_demo.py', 'paint_benchmark.py']}
    manifest = {'version': 'shared-texture-fresh-v1', 'options': asdict(OPTIONS),
        'corruption_cases': CASES, 'sampler_seeds': SAMPLER_SEEDS, 'designs': DESIGNS,
        'image_size': 128, 'geometry_resolution': 56, 'held_out': [vars(c) for c in HELD_OUT],
        'source_sha256': hashes,
        'protocol': 'Written before case generation; all listed runs reported; no hyperparameter changes between cases.'}
    (directory/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    all_runs = []
    start = time.perf_counter()
    train_cameras = [Camera(a, 8) for a in range(0, 360, 45)]+[Camera(20, 65)]
    for design in DESIGNS:
        for data_seed, amount in CASES:
            case_name = f'{design}-{data_seed}'
            case = directory/case_name
            case.mkdir(exist_ok=True)
            rng = np.random.default_rng(data_seed)
            entries = []
            for i, camera in enumerate(train_cameras):
                truth, mask = trouser_truth(camera, 128, design)
                art = inconsistent_paint(truth, mask, rng, amount, conflict=i % 3 == 1)
                if design == 'patchwork':
                    # Unmodelled low-frequency painted tone is an intentional stressor.
                    v, u = np.mgrid[:128, :128]/128
                    tone = 1+.08*np.sin(2*np.pi*u+rng.uniform(0, 2*np.pi))*np.cos(np.pi*v)
                    art[mask] = np.clip(art[mask]*tone[mask, None], 0, 1)
                name = f'input-{i}.png'
                save_image(case/name, art, mask)
                entries.append({'image': name, 'camera': vars(camera)})
            (case/'project.json').write_text(json.dumps({'views': entries}, indent=2)+'\n')
            views, _ = load_project(case/'project.json')
            baseline = reconstruct(views, 56)
            bake_paint(baseline, views, OPTIONS.texture_resolution)
            # Clean truth is constructed only for evaluation, after observed project creation.
            truth_views = [trouser_truth(camera, 128, design) for camera in HELD_OUT]
            baseline_renders = [render(baseline, camera, 128, False) for camera in HELD_OUT]
            baseline_metrics = []
            for (truth, mask), (pred, pmask, _) in zip(truth_views, baseline_renders):
                row = metrics(pred, pmask, truth, mask)
                row['gradient_mae'] = detail_error(pred, truth, pmask & mask)
                baseline_metrics.append(row)
            for sampler_seed in SAMPLER_SEEDS:
                record = {'case': case_name, 'design': design, 'data_seed': data_seed,
                          'sampler_seed': sampler_seed, 'distortion_fraction': amount,
                          'baseline': baseline_metrics, 'methods': {}}
                predictions = {}
                for objective in ['pairwise', 'shared']:
                    model = Model(baseline.sdf.copy(), baseline.color.copy(), dict(baseline.metadata))
                    options = replace(OPTIONS, seed=sampler_seed, objective=objective)
                    fit = fit_paint(model, views, options)
                    evaluated, images = [], []
                    for camera, (truth, mask) in zip(HELD_OUT, truth_views):
                        pred, pmask, _ = render(model, camera, 128, False)
                        row = metrics(pred, pmask, truth, mask)
                        row['gradient_mae'] = detail_error(pred, truth, pmask & mask)
                        evaluated.append(row)
                        images.append(pred)
                    record['methods'][objective] = {'held_out': evaluated,
                        'fit_seconds': fit['seconds'], 'initial_energy': fit['initial_energy'],
                        'selected_energy': fit['map_energy'],
                        'hypothesis_energies': fit['hypothesis_energies'],
                        'export_energy': fit.get('export_energy'),
                        'texture_updates_monotone': all(t['after'] <= t['before']+1e-10
                                                       for t in fit.get('texture_updates', []))}
                    predictions[objective] = images
                    print(f"{case_name} / sampler {sampler_seed} / {objective}: "
                          f"MAE {np.mean([r['overlap_color_mae'] for r in evaluated]):.4f}", flush=True)
                # Save every run at the same predeclared camera, rather than selecting a winner.
                panel(case/f'comparison-seed-{sampler_seed}.png', [
                    ('Clean unseen view', truth_views[0][0]), ('Projection baseline', baseline_renders[0][0]),
                    ('Fixed pairwise', predictions['pairwise'][0]), ('Shared texture', predictions['shared'][0])])
                all_runs.append(record)
                (directory/'runs.json').write_text(json.dumps(all_runs, indent=2)+'\n')
    summary = summarize(all_runs)
    summary['seconds'] = time.perf_counter()-start
    (directory/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def summarize(runs):
    result = {'run_count': len(runs), 'held_out_cameras_per_run': len(HELD_OUT), 'metrics': {}}
    for method in ['baseline', 'pairwise', 'shared']:
        rows = [row for run in runs for row in
                (run['baseline'] if method == 'baseline' else run['methods'][method]['held_out'])]
        result['metrics'][method] = {key: float(np.mean([row[key] for row in rows])) for key in
            ['overlap_color_mae', 'foreground_psnr_db', 'gradient_mae']}
    for other in ['baseline', 'pairwise']:
        gains = []
        for run in runs:
            shared = np.mean([r['overlap_color_mae'] for r in run['methods']['shared']['held_out']])
            rows = run['baseline'] if other == 'baseline' else run['methods'][other]['held_out']
            reference = np.mean([r['overlap_color_mae'] for r in rows])
            gains.append(float(1-shared/reference))
        result[f'shared_vs_{other}'] = {'improved_runs': int(np.sum(np.array(gains) > 0)),
            'total_runs': len(runs), 'relative_mae_gains': gains,
            'worst_gain': min(gains), 'best_gain': max(gains)}
    return result
