"""Checks of probabilistic registration using independently imposed image shifts."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy import ndimage as ndi

from desina.core import Camera, Model, View, brush, grid, render
from desina.paint import (PaintOptions, displacement, fit_paint, fuse, regularization,
                          residual_cost, warp_basis)


class PaintTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        xyz = grid(40)
        sdf = (np.linalg.norm(xyz, axis=-1)-.65).astype(np.float32)
        color = np.full((*sdf.shape, 3), [.1, .2, .3], np.float32)
        color[np.sin(xyz[..., 0]*28)+np.cos(xyz[..., 1]*22) > .8] = [.9, .7, .3]
        truth = Model(sdf, color)
        rgb, mask, _ = render(truth, Camera(), 80, False)
        shifted = ndi.shift(rgb, (0, 3, 0), order=1, mode='nearest')
        cls.views = [View(Camera(), rgb, mask), View(Camera(), shifted, mask)]
        cls.model = Model(sdf.copy(), color.copy())
        cls.options = PaintOptions(seed=19, steps=180, chains=1, samples=600, reference=0,
                                   texture_resolution=64, posterior_draws=4)
        cls.report = fit_paint(cls.model, cls.views, cls.options)
        cls.original_sdf = sdf

    def test_known_shift_is_recovered_without_geometry_changes(self):
        self.assertLess(self.report['map_energy'], self.report['initial_energy']*.35)
        shift = displacement(self.model.paint_warps[1], warp_basis(np.array([[.5, .5]]), 4))[0]
        self.assertAlmostEqual(float(shift[0]), 3/80, delta=.014)
        self.assertAlmostEqual(float(shift[1]), 0, delta=.014)
        np.testing.assert_array_equal(self.model.sdf, self.original_sdf)
        np.testing.assert_array_equal(self.model.paint_warps[0], 0)
        self.assertGreater(np.std(self.model.paint_samples[:, 1]), 0)
        self.assertEqual(self.model.uncertainty.shape, (64,)*3)
        self.assertTrue(np.isfinite(self.model.color).all())
        self.assertTrue(np.all((self.model.uncertainty >= 0) & (self.model.uncertainty <= 1)))

    def test_saved_fit_and_high_resolution_paint(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'fit.npz'
            self.model.save(path)
            loaded = Model.load(path)
            for key in ['sdf', 'color', 'uncertainty', 'paint_warps', 'paint_samples']:
                np.testing.assert_array_equal(getattr(loaded, key), getattr(self.model, key))
            point = np.array([[0., 0., .65]])
            before = loaded.sample(point, loaded.color)[0]
            brush(loaded, point[0], .16, 'paint', [1, 0, 0])
            self.assertGreater(loaded.sample(point, loaded.color)[0, 0], before[0])
            self.assertEqual(loaded.color.shape, (64, 64, 64, 3))
            self.assertEqual(loaded.sdf.shape, (40,)*3)
            # Rendering remains independent of source views / registration state.
            rgb, mask, _ = render(loaded, Camera(23, 12), 48, False)
            loaded.paint_warps = loaded.paint_samples = None
            again, _, _ = render(loaded, Camera(23, 12), 48, False)
            np.testing.assert_array_equal(again, rgb)
            self.assertTrue(mask.any())

    def test_seeded_sampling_is_reproducible(self):
        options = PaintOptions(seed=33, steps=8, chains=1, samples=100, reference=0,
                               texture_resolution=24, posterior_draws=2)
        models = []
        for _ in range(2):
            model = Model(self.original_sdf.copy(), self.model.color.copy())
            fit_paint(model, self.views, options)
            models.append(model)
        np.testing.assert_array_equal(models[0].color, models[1].color)
        np.testing.assert_array_equal(models[0].paint_samples, models[1].paint_samples)

    def test_mixture_mode_is_not_a_color_average(self):
        points = np.array([[i*.02, 0, .5] for i in range(12)])
        # Two observations support cream, one says blue: retain a mode, not teal.
        colors = np.tile(np.array([[[.9, .7, .5]], [[.9, .7, .5]], [[.1, .2, .4]]]), (1, 12, 1))
        selected, uncertainty, _ = fuse(points, colors, np.ones((3, 12)), PaintOptions())
        np.testing.assert_allclose(selected, np.tile([.9, .7, .5], (12, 1)))
        self.assertTrue((uncertainty > 0).all())
        # Uniform outlier likelihood keeps arbitrary disagreement finite.
        cost = residual_cost(np.zeros((1, 3)), np.ones((1, 3)), .065, .12)
        self.assertTrue(np.isfinite(cost).all())

    def test_constraints_and_invisible_overlap_fail_cleanly(self):
        options = PaintOptions()
        with self.assertRaises(ValueError):
            PaintOptions(max_warp=float('nan')).validate(2)
        with self.assertRaises(ValueError):
            PaintOptions(samples=0).validate(2)
        huge = np.ones((4, 4, 2))*.2
        self.assertFalse(np.isfinite(regularization(huge, options)))
        # Opposite cameras have no jointly visible points on this convex sphere.
        back_rgb, back_mask, _ = render(Model(self.original_sdf, self.model.color), Camera(180), 80, False)
        with self.assertRaisesRegex(ValueError, 'visible|overlap'):
            fit_paint(Model(self.original_sdf.copy(), self.model.color.copy()),
                [self.views[0], View(Camera(180), back_rgb, back_mask)],
                PaintOptions(steps=1, chains=1, samples=100, texture_resolution=24))


class ReviewRegressionTests(unittest.TestCase):
    def test_source_updates_decrease_energy_without_oscillation(self):
        from desina.paint import smooth_sources
        points = np.array([[0, 0, .5], [.02, 0, .5]])
        unary = np.array([[0., .025], [.025, 0.]])
        source, history = smooth_sources(unary, points, sweeps=10)
        self.assertEqual(source[0], source[1])
        self.assertTrue(np.all(np.diff(history) <= 1e-12))
        self.assertLess(history[-1], history[0])

    def test_best_export_includes_every_tail_move(self):
        from unittest.mock import patch
        from desina.paint import register
        class Quadratic:
            views = [None, None]
            images = [np.zeros((4, 4, 3))]*2
            def texture_state(self): return None
            def restore_texture(self, state): pass
            def refine_texture(self, controls, observations): return False
            def observe(self, i, controls): return np.array([controls.mean()])
            def observations(self, controls): return np.array([self.observe(i, c) for i, c in enumerate(controls)])
            def local_loss(self, i, controls, observation, observations):
                return (controls.mean()-(.04 if i else 0))**2
            def energy(self, controls, observations):
                return float(np.mean([self.local_loss(i, c, None, None) for i, c in enumerate(controls)]))
        reg = Quadratic()
        options = PaintOptions(steps=1, chains=1, reference=0, posterior_draws=2, objective='pairwise')
        with patch('desina.paint.propose', side_effect=lambda c, rng, scale: c+.001):
            best, draws, report = register(reg, options)
        self.assertGreater(best[1].mean(), .003)
        self.assertLessEqual(report['map_energy'], min(reg.energy(d, None) for d in draws)+1e-12)
        self.assertAlmostEqual(report['map_energy'], reg.energy(best, None), places=12)
        self.assertLess(report['map_energy'], report['chains'][0]['pre_tail_energy'])

    def test_shared_objective_matches_export_and_conditional_delta(self):
        from desina.paint import prepare_surface, prepare_views
        from desina.paint_joint import SharedTextureRegistration
        xyz = grid(32)
        sdf = np.linalg.norm(xyz, axis=-1)-.65
        color = np.full((*sdf.shape, 3), [.15, .3, .5], np.float32)
        color[xyz[..., 1] > .12] = [.8, .6, .2]
        model = Model(sdf.astype(np.float32), color)
        rgb, mask, _ = render(model, Camera(), 64, False)
        views = [View(Camera(), rgb, mask), View(Camera(25), *render(model, Camera(25), 64, False)[:2])]
        options = PaintOptions(steps=12, chains=2, samples=180, texture_resolution=40, posterior_draws=2)
        report = fit_paint(model, views, options)
        shell, indices, points, normals = prepare_surface(model, options.texture_resolution)
        uv, vis, images = prepare_views(model, views, points, normals)
        eligible = np.flatnonzero((vis > .015).sum(axis=0) >= 2)
        chosen = np.random.default_rng(options.seed).choice(eligible, min(len(eligible), options.samples), replace=False)
        reg = SharedTextureRegistration(model, views, uv[:, chosen], vis[:, chosen], images, options,
            points[chosen], uv, vis, points, indices, shell)
        reg.sigma = .065
        obs = reg.observations(model.paint_warps)
        energy = reg.energy(model.paint_warps, obs)
        self.assertAlmostEqual(energy, report['export_energy'], places=6)
        self.assertAlmostEqual(energy, report['map_energy'], places=6)
        np.testing.assert_array_equal(reg.texture_state(), model.color)
        self.assertTrue(all(x['after'] <= x['before']+1e-12 for x in report['texture_updates']))
        self.assertGreaterEqual(len(model.paint_hypothesis_colors), 2)
        self.assertEqual(len(model.paint_samples), options.chains*options.posterior_draws)
        candidate = model.paint_warps.copy()
        candidate[0] += .0001
        updated = reg.observations(candidate)
        local = (reg.local_loss(0, candidate[0], updated[0], obs)-
                 reg.local_loss(0, model.paint_warps[0], obs[0], obs))/len(views)
        self.assertAlmostEqual(local, reg.energy(candidate, updated)-energy, places=10)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'hypotheses.npz'
            model.save(path)
            loaded = Model.load(path)
            for key in ['paint_hypothesis_colors', 'paint_hypothesis_warps', 'paint_hypothesis_energies']:
                np.testing.assert_array_equal(getattr(model, key), getattr(loaded, key))


if __name__ == '__main__':
    unittest.main()
