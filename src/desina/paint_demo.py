"""Inconsistent painted trousers: independent analytic truth, RGB-only perturbations.

No feature masks, true correspondence fields, or clean textures enter the fitter.
Tear-like regions are surface paint in this experiment, not physical cloth holes.
"""
import json
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from PIL import Image

from .core import Camera, Model, bake_paint, load_project, reconstruct, render, save_image
from .demo import metrics, panel
from .paint import PaintOptions, fit_paint

PARTS = [((-.235, -.13, 0), (.245, .69, .29)),
         ((.235, -.13, 0), (.245, .69, .29)),
         ((0, .43, 0), (.47, .24, .285))]


def trouser_truth(camera, size, design='rips'):
    origins, direction = camera.rays(size)
    best = np.full(len(origins), np.inf)
    material = np.zeros(len(origins), int)
    for i, (center, radius) in enumerate(PARTS):
        o, d = (origins-center)/radius, direction/np.array(radius)
        a, b = d @ d, 2*(o @ d)
        disc = b*b-4*a*(np.sum(o*o, axis=-1)-1)
        t = (-b-np.sqrt(np.maximum(disc, 0)))/(2*a)
        use = (disc >= 0) & (t > 0) & (t < best)
        best[use], material[use] = t[use], i
    mask = np.isfinite(best)
    p = origins[mask]+best[mask, None]*direction
    part = material[mask]
    center_x = np.where(part == 0, -.235, np.where(part == 1, .235, 0))
    theta = np.arctan2(p[:, 2], p[:, 0]-center_x)
    y = p[:, 1]
    # Fine plaid with continuous variation; no named feature IDs survive this function.
    plaid = .018*np.sin(theta*22)*np.cos(y*70)
    colors = np.tile([.13, .27, .43], (len(p), 1))+plaid[:, None]
    seam = (np.abs(np.sin(theta)) < .09) & (part != 2)
    stitches = seam & (np.sin(y*110) > -.1)
    colors[stitches] = [.93, .69, .26]
    # Ragged tear-like paint: cream centers, pale fraying, dark borders.
    distance = np.full(len(p), 10.)
    for angle, height, width in [(1.30, -.10, .52), (1.65, -.46, .42),
                                  (-1.35, -.28, .45), (.1, -.52, .32)]:
        h = height + np.where(part == 1, .10, 0)
        dtheta = np.arctan2(np.sin(theta-angle), np.cos(theta-angle))
        rag = .012*np.sin(theta*51+y*20)+.008*np.sin(theta*87)
        d = np.maximum(np.abs(dtheta)/width, np.abs(y-h)/(.045+rag))
        distance = np.minimum(distance, np.where(part < 2, d, 10))
    colors[distance < 1.35] = [.045, .08, .13]
    colors[distance < 1.15] = [.75, .82, .83]
    colors[distance < .82] = [.88, .66, .46]
    threads = (distance < 1.15) & (np.sin(y*380) > .60)
    colors[threads] = [.90, .88, .74]
    waistband = part == 2
    colors[waistband] = np.array([.10, .20, .32])+plaid[waistband, None]
    belt = waistband & (y > .48) & (y < .53)
    colors[belt] = [.78, .52, .20]
    if design == 'patchwork':
        palette = np.array([[.12, .39, .31], [.85, .56, .19], [.65, .20, .27], [.88, .82, .62]])
        row = np.floor((y+.85)*6).astype(int)
        column = np.floor((theta+np.pi)*2.5).astype(int)
        colors = palette[(row+2*column+part) % len(palette)]
        stitches = (np.abs(np.sin((y+.85)*6*np.pi)) < .10) | \
                   (np.abs(np.sin((theta+np.pi)*2.5*np.pi)) < .09)
        colors[stitches] = [.055, .09, .10]
        colors += (.012*np.sin(theta*45)*np.cos(y*100))[:, None]
    elif design != 'rips':
        raise ValueError('Unknown procedural paint design.')
    rgb = np.full((len(origins), 3), .96)
    rgb[mask] = np.clip(colors, 0, 1)
    return rgb.reshape(size, size, 3), mask.reshape(size, size)


def inconsistent_paint(rgb, mask, rng, amount=.035, conflict=True):
    """Analytic sinusoidal warps, different from the fitter's control-grid family."""
    n = len(mask)
    v, u = np.mgrid[:n, :n]/n
    phase = rng.uniform(0, 2*np.pi, 4)
    # All views are imperfect, including the arbitrarily chosen reference view.
    du = amount*(.7*np.sin(2*np.pi*v+phase[0])+.3*np.cos(4*np.pi*u+phase[1]))
    dv = amount*(.7*np.cos(2*np.pi*u+phase[2])+.3*np.sin(3*np.pi*v+phase[3]))
    nearest = ndi.distance_transform_edt(~mask, return_distances=False, return_indices=True)
    extended = rgb[tuple(nearest)]
    coords = [(v+dv)*n, (u+du)*n]
    art = np.stack([ndi.map_coordinates(extended[..., c], coords, order=1, mode='nearest')
                    for c in range(3)], -1)
    # Deliberately unmatched short strokes; these cannot be fixed with any smooth warp.
    if conflict:
        cx, cy = rng.uniform(.35, .65), rng.uniform(.42, .68)
        extra = (np.abs(u-cx) < .065) & (np.abs(v-cy+.1*(u-cx)) < .009) & mask
        art[extra] = [.88, .65, .45]
    art = np.clip(art*rng.uniform(.97, 1.03, 3)+rng.uniform(-.008, .008, 3), 0, 1)
    art[~mask] = .96
    return art


def run_paint_demo(directory, resolution=80, size=160, options=None, amount=.035, turntable=True):
    if not 48 <= size <= 512 or not np.isfinite(amount) or not 0 <= amount <= .08:
        raise ValueError('Paint demo needs size 48..512 and distortion in [0,.08].')
    options = options or PaintOptions()
    options.validate(9)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    cameras = [Camera(a, 8) for a in range(0, 360, 45)]+[Camera(20, 65)]
    rng = np.random.default_rng(2026)
    entries = []
    previews = []
    for i, camera in enumerate(cameras):
        clean, mask = trouser_truth(camera, size)
        art = inconsistent_paint(clean, mask, rng, amount, conflict=(i % 3 == 1))
        save_image(directory/f'input-{i}.png', art, mask)
        entries.append({'image': f'input-{i}.png', 'camera': vars(camera)})
        if i in (0, 1, 2):
            previews.append((f'Input {i} / independently distorted', art))
    (directory/'project.json').write_text(json.dumps({'views': entries,
        'description': 'Analytic trousers; silhouettes calibrated; paint independently distorted.'}, indent=2)+'\n')
    # Only project.json, observed pixels, camera hints, and reconstructed SDF go onward.
    views, _ = load_project(directory/'project.json')
    baseline = reconstruct(views, resolution)
    bake_paint(baseline, views, options.texture_resolution)
    baseline.save(directory/'baseline.npz')
    fitted = Model(baseline.sdf.copy(), baseline.color.copy(), dict(baseline.metadata))
    fit_paint(fitted, views, options, progress=lambda line: print(line, flush=True))
    fitted.save(directory/'fitted.npz')
    # Isolate gains from robust source fusion versus correspondence registration.
    fusion = Model(baseline.sdf.copy(), baseline.color.copy(), dict(baseline.metadata))
    fit_paint(fusion, views, options, align=False)
    fusion.save(directory/'fusion-only.npz')
    panel(directory/'inputs.png', previews, 'Only distorted images and camera hints are given to the fitter.')
    results = []
    for k, camera in enumerate([Camera(23, 18), Camera(113, -8), Camera(248, 25)]):
        truth, mask = trouser_truth(camera, size)
        row = {'camera': vars(camera)}
        panels = [('Unseen camera / clean truth', truth)]
        for name, model in [('baseline', baseline), ('fusion_only', fusion), ('monte_carlo', fitted)]:
            pred, pmask, points = render(model, camera, size, False)
            row[name] = metrics(pred, pmask, truth, mask)
            save_image(directory/f'heldout-{k}-{name}.png', pred)
            panels.append((name.replace('_', ' '), pred))
        save_image(directory/f'heldout-{k}-truth.png', truth)
        panel(directory/f'comparison-{k}.png', panels,
            f"Foreground PSNR: baseline {row['baseline']['foreground_psnr_db']:.2f} / "
            f"fusion {row['fusion_only']['foreground_psnr_db']:.2f} / MC {row['monte_carlo']['foreground_psnr_db']:.2f} dB")
        if k == 0:
            heat = np.full_like(truth, .96)
            heat[pmask] = uncertainty_color(fitted.sample(points[pmask], fitted.uncertainty))
            save_image(directory/'uncertainty.png', heat)
        results.append(row)
    scores = {name: float(np.mean([row[name]['foreground_psnr_db'] for row in results]))
              for name in ['baseline', 'fusion_only', 'monte_carlo']}
    report = {'distortion_fraction': amount, 'size': size, 'resolution': resolution,
              'input_views': len(views), 'held_out': results, 'mean_foreground_psnr_db': scores,
              'fit': fitted.metadata['paint_fit'],
              'scope': 'Fixed reconstructed geometry; tear-like regions are paint, not physical holes.'}
    (directory/'metrics.json').write_text(json.dumps(report, indent=2)+'\n')
    if turntable:
        frames = []
        for angle in range(0, 360, 15):
            parts = []
            for model in (baseline, fitted):
                rgb, _, _ = render(model, Camera(angle, 15), 128, False)
                parts.append(rgb)
            frames.append(Image.fromarray(np.uint8(np.concatenate(parts, axis=1)*255)))
        frames[0].save(directory/'turntable.gif', save_all=True, append_images=frames[1:], duration=120, loop=0)
    print(json.dumps({'mean_foreground_psnr_db': scores, 'fit_seconds': report['fit']['seconds']}, indent=2))
    return report


def uncertainty_color(value):
    value = np.clip(value, 0, 1)
    return np.stack([value, .25+.50*(1-value), 1-value], axis=-1)
