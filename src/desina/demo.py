"""Independent analytic ellipsoid renderer, used ONLY to generate/evaluate the demo."""
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw
from .core import Camera, load_project, reconstruct, render, save_image, brush

# This scene never enters the reconstruction code or saved reconstruction project.
PARTS = [
    ((0, -.12, 0), (.40, .52, .32), (.13, .58, .67)),
    ((0, .47, .02), (.30, .29, .28), (.16, .65, .73)),
    ((-.39, -.10, -.01), (.16, .32, .18), (.08, .38, .51)),
    ((.39, -.10, -.01), (.16, .32, .18), (.08, .38, .51)),
    ((0, .36, .34), (.15, .10, .21), (.98, .60, .15)),
    ((-.18, -.65, .10), (.16, .08, .20), (.98, .60, .15)),
    ((.18, -.65, .10), (.16, .08, .20), (.98, .60, .15)),
    ((-.16, .51, .255), (.045, .055, .025), (.025, .035, .05)),
    ((.16, .51, .255), (.045, .055, .025), (.025, .035, .05)),
    ((0, .76, -.025), (.085, .15, .075), (.97, .40, .25)),
]


def ground_truth(camera, size):
    origins, direction = camera.rays(size)
    best = np.full(len(origins), np.inf)
    material = np.zeros(len(origins), int)
    for i, (center, radius, _) in enumerate(PARTS):
        o = (origins-center)/radius
        d = direction/np.array(radius)
        a = d @ d
        b = 2*(o @ d)
        c = np.sum(o*o, -1)-1
        disc = b*b-4*a*c
        t = (-b-np.sqrt(np.maximum(disc, 0)))/(2*a)
        use = (disc >= 0) & (t > 0) & (t < best)
        best[use], material[use] = t[use], i
    mask = np.isfinite(best)
    p = origins[mask]+best[mask, None]*direction
    rgb = np.full((len(origins), 3), .96)
    colors = np.array([part[2] for part in PARTS])[material[mask]]
    # A painted belly and spots belong to the surface, not to the camera.
    belly = (material[mask] == 0) & (p[:, 2] > .21) & (np.abs(p[:, 0]) < .27)
    colors[belly] = [.94, .88, .68]
    spots = (material[mask] == 0) & (p[:, 2] < -.2) & \
        (np.sin(p[:, 0]*26)*np.cos(p[:, 1]*26) > .65)
    colors[spots] = [.95, .72, .24]
    rgb[mask] = colors
    return rgb.reshape(size, size, 3), mask.reshape(size, size)


def metrics(pred, pmask, target, tmask):
    overlap = pmask & tmask
    union = pmask | tmask
    mse = float(np.mean((pred[union]-target[union])**2))
    return {'silhouette_iou': float(overlap.sum()/max(union.sum(), 1)),
            'foreground_psnr_db': float(-10*np.log10(max(mse, 1e-10))),
            'overlap_color_mae': float(np.abs(pred[overlap]-target[overlap]).mean())}


def panel(path, panels, footer=''):
    tile = 256
    canvas = Image.new('RGB', (tile*len(panels), tile+76), '#f5f5f5')
    draw = ImageDraw.Draw(canvas)
    for i, (label, rgb) in enumerate(panels):
        img = Image.fromarray(np.uint8(np.clip(rgb, 0, 1)*255)).resize((tile, tile))
        canvas.paste(img, (i*tile, 30))
        draw.text((i*tile+12, 10), label, fill='#172a32')
    draw.text((12, tile+42), footer, fill='#172a32')
    canvas.save(path)


def run_demo(directory, resolution=80, size=160):
    if not 16 <= size <= 1024:
        raise ValueError('Image size must be between 16 and 1024.')
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    entries = []
    for i, camera in enumerate([Camera(a, 5) for a in range(0, 360, 60)] + [Camera(35, 65)]):
        rgb, mask = ground_truth(camera, size)
        name = f'input-{i}.png'
        save_image(directory/name, rgb, mask)
        entries.append({'image': name, 'camera': vars(camera)})
    spec = {'description': 'Synthetic painted bird; orthographic, registered silhouettes.',
            'views': entries}
    project = directory/'project.json'
    project.write_text(json.dumps(spec, indent=2)+'\n')
    # Reconstruction has access only to these 7 raster inputs and camera hints.
    start = time.perf_counter()
    views, _ = load_project(project)
    model = reconstruct(views, resolution)
    seconds = time.perf_counter()-start
    model.save(directory/'bird.npz')
    scores = []
    for i, camera in enumerate([Camera(30, 22), Camera(145, -12), Camera(275, 35)]):
        target, tmask = ground_truth(camera, size)
        pred, pmask, _ = render(model, camera, size, shaded=False)
        score = {'camera': vars(camera), **metrics(pred, pmask, target, tmask)}
        scores.append(score)
        save_image(directory/f'heldout-{i}-truth.png', target)
        save_image(directory/f'heldout-{i}-reconstruction.png', pred)
        error = np.ones_like(pred)*.96
        error[pmask | tmask] = np.abs(pred-target)[pmask | tmask]
        panel(directory/f'comparison-{i}.png', [('Unseen camera / truth', target),
              ('Reconstruction / fixed 3D paint', pred), ('Absolute RGB difference', error)],
              f"IoU {score['silhouette_iou']:.3f}  |  foreground PSNR {score['foreground_psnr_db']:.1f} dB")
    shaded, _, _ = render(model, Camera(30, 22), size)
    save_image(directory/'shaded.png', shaded)
    # Demonstrate persistent 3D editing: a paint stroke stays on the same surface as it rotates.
    brush(model, [0, -.10, .32], .22, 'paint', [.85, .12, .36])
    model.save(directory/'bird-painted.npz')
    edited_a, _, _ = render(model, Camera(30, 22), size, False)
    edited_b, _, _ = render(model, Camera(-30, 22), size, False)
    panel(directory/'editing.png', [('Before', render_model(directory/'bird.npz', Camera(30, 22), size)),
          ('3D paint stroke', edited_a), ('Same stroke, rotated', edited_b)])
    frames = []
    for a in range(0, 360, 15):
        rgb, _, _ = render(model, Camera(a, 16), 128)
        frames.append(Image.fromarray(np.uint8(rgb*255)))
    frames[0].save(directory/'turntable.gif', save_all=True, append_images=frames[1:], duration=100, loop=0)
    result = {'resolution': resolution, 'input_views': len(views), 'image_size': size,
              'reconstruction_seconds': seconds, 'paint_coverage': model.metadata['paint_coverage'],
              'held_out': scores, 'mean_iou': float(np.mean([s['silhouette_iou'] for s in scores]))}
    (directory/'metrics.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
    return result


def render_model(path, camera, size):
    from .core import Model
    return render(Model.load(path), camera, size, False)[0]
