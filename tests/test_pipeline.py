"""Behavioral checks against analytic geometry and genuinely withheld camera views."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from desina.core import (Camera, Model, View, brush, grid, load_project,
                         reconstruct, render, trace)
from desina.demo import ground_truth, metrics


class PipelineTests(unittest.TestCase):
    def test_sphere_first_hit_and_empty_rays(self):
        xyz = grid(48)
        sdf = np.linalg.norm(xyz, axis=-1)-.6
        model = Model(sdf, np.ones((*sdf.shape, 3), np.float32)*.4)
        points, mask = trace(model, Camera(37, 21), 65)
        self.assertTrue(mask[32, 32])
        self.assertFalse(mask[0, 0])
        self.assertLess(np.abs(np.linalg.norm(points[mask], axis=-1)-.6).max(), .003)
        self.assertGreater(float(points[32, 32] @ Camera(37, 21).basis()[2]), .59)

    def test_held_out_bird_and_visibility(self):
        cameras = [Camera(a, 5) for a in range(0, 360, 60)] + [Camera(35, 65)]
        views = [View(camera, *ground_truth(camera, 96)) for camera in cameras]
        model = reconstruct(views, 48)
        novel = Camera(30, 22)
        self.assertNotIn(vars(novel), [vars(c) for c in cameras])
        target, tmask = ground_truth(novel, 96)
        pred, pmask, _ = render(model, novel, 96, False)
        score = metrics(pred, pmask, target, tmask)
        self.assertGreater(score['silhouette_iou'], .85)
        self.assertLess(score['overlap_color_mae'], .09)
        # Front cream belly and rear turquoise must not be averaged through the volume.
        front = model.sample(np.array([[0, -.1, .32]]), model.color)[0]
        back = model.sample(np.array([[0, -.1, -.32]]), model.color)[0]
        self.assertGreater(front[0]-back[0], .4)

    def test_persistent_edit_and_sculpt(self):
        xyz = grid(40)
        sdf = (np.linalg.norm(xyz, axis=-1)-.55).astype(np.float32)
        model = Model(sdf, np.zeros((*sdf.shape, 3), np.float32), {'edits': []})
        before = model.sdf.copy()
        brush(model, [0, 0, .55], .2, 'paint', [1, 0, 0])
        np.testing.assert_array_equal(before, model.sdf)
        self.assertGreater(model.sample(np.array([[0, 0, .55]]), model.color)[0, 0], .7)
        self.assertEqual(model.sample(np.array([[0, 0, -.55]]), model.color)[0, 0], 0)
        # Same world-space mark is visible in two independently traced camera views.
        for angle in [-30, 30]:
            rgb, mask, _ = render(model, Camera(angle), 65, False)
            self.assertGreater(rgb[mask, 0].max(), .65)
        brush(model, [.58, 0, 0], .2, 'add')
        self.assertLess(model.sample(np.array([[.70, 0, 0]]))[0], 0)
        brush(model, [0, 0, .50], .22, 'subtract')
        self.assertGreater(model.sample(np.array([[0, 0, .50]]))[0], 0)
        brush(model, [.58, 0, 0], .2, 'smooth')
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'edited.npz'
            model.save(path)
            loaded = Model.load(path)
            np.testing.assert_array_equal(loaded.sdf, model.sdf)
            np.testing.assert_array_equal(loaded.color, model.color)
            self.assertEqual(loaded.metadata, model.metadata)

    def test_drawing_masks_and_project_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            outline = np.full((32, 32, 3), 255, np.uint8)
            outline[7:25, 7] = outline[7:25, 24] = 0
            outline[7, 7:25] = outline[24, 7:25] = 0
            Image.fromarray(outline).save(tmp/'drawing.png')
            spec = {'views': [{'image': 'drawing.png', 'camera': {'azimuth': a}} for a in [0, 90]]}
            (tmp/'project.json').write_text(json.dumps(spec))
            views, _ = load_project(tmp/'project.json')
            self.assertTrue(views[0].mask[16, 16])
            self.assertFalse(views[0].mask[0, 0])
            # Explicit masks allow white paint to be distinguished from white background.
            Image.fromarray(np.uint8(views[0].mask)*255).save(tmp/'mask.png')
            for entry in spec['views']:
                entry['mask'] = 'mask.png'
            (tmp/'project.json').write_text(json.dumps(spec))
            explicit, _ = load_project(tmp/'project.json')
            np.testing.assert_array_equal(explicit[0].mask, views[0].mask)
            spec['views'][0]['camera']['scale'] = 0
            (tmp/'project.json').write_text(json.dumps(spec))
            with self.assertRaises(ValueError):
                load_project(tmp/'project.json')

    def test_invalid_inputs_fail(self):
        sdf = np.ones((24,)*3, np.float32)
        model = Model(sdf, np.zeros((*sdf.shape, 3), np.float32))
        with self.assertRaises(ValueError):
            brush(model, [0, 0, 0], 0)
        with self.assertRaises(ValueError):
            brush(model, [0, 0, 0], mode='unknown')
        with self.assertRaises(ValueError):
            reconstruct([], 8)


if __name__ == '__main__':
    unittest.main()
