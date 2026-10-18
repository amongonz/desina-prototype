"""Shared 3D texture likelihood with conditional Monte Carlo warp updates.

The data term samples the actual RGB volume with the renderer's interpolation.
Full-surface color-mode/IRLS estimates are proposals, not a separate output stage:
only proposals decreasing the same joint objective are accepted.
"""
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

from .paint import Registration, displacement, residual_cost, sample_image, warp_basis


class SharedTextureRegistration(Registration):
    def __init__(self, model, views, uv, visibility, images, options, points,
                 full_uv, full_visibility, full_points, indices, shell):
        super().__init__(views, uv, visibility, images, options)
        self.model, self.points = model, points
        self.full_uv, self.full_visibility, self.full_points = full_uv, full_visibility, full_points
        self.indices, self.shell = indices, shell
        self.full_bases = [warp_basis(x, options.knots) for x in full_uv]
        self.nearest = ndi.distance_transform_edt(~shell, return_distances=False, return_indices=True)
        self.supported = np.any(full_visibility >= .001, axis=0)
        if not self.supported.any():
            raise ValueError('No visible surface paint.')
        self.missing_nearest = cKDTree(full_points[self.supported]).query(full_points[~self.supported])[1]
        self.total_weight = max(visibility.sum(), 1e-8)
        self.neighbors = cKDTree(points).query(points, k=min(5, len(points)))[1][:, 1:]
        self.texture_updates = []
        initial = model.color
        if len(initial) != options.texture_resolution:
            ratio = options.texture_resolution/len(initial)
            initial = ndi.zoom(initial, (ratio, ratio, ratio, 1), order=1)
        self.restore_texture(initial)

    def texture_state(self):
        # Arrays are immutable throughout optimization. Retain references without
        # copying every RGB cube on every accepted warp proposal.
        return self.texture

    def restore_texture(self, state):
        self.texture = state
        self.prediction = self.model.sample(self.points, self.texture)
        if self.neighbors.size:
            difference = self.prediction[:, None]-self.prediction[self.neighbors]
            self.color_prior = self.options.texture_prior*float(np.mean(
                np.sqrt(np.mean(difference*difference, axis=-1)+.015**2)-.015))
        else:
            self.color_prior = 0.

    def local_loss(self, i, controls, observation, all_observations):
        data = np.sum(self.visibility[i]*residual_cost(observation, self.prediction,
                          self.sigma, self.options.outlier_rate))*len(self.views)/self.total_weight
        return data+self.prior_loss(i, controls)

    def energy(self, controls, observations):
        data = np.sum(self.visibility*residual_cost(observations, self.prediction[None],
                          self.sigma, self.options.outlier_rate))/self.total_weight
        prior = np.mean([self.prior_loss(i, controls[i]) for i in range(len(self.views))])
        return float(data+prior+self.color_prior)

    def volume_from_surface(self, surface):
        surface = surface.copy()
        surface[~self.supported] = surface[self.supported][self.missing_nearest]
        volume = np.zeros((*self.shell.shape, 3), np.float32)
        volume[tuple(self.indices.T)] = surface
        return volume[tuple(self.nearest)]

    def refine_texture(self, controls, observations):
        before = self.energy(controls, observations)
        old = self.texture
        current = self.model.sample(self.full_points, self.texture)
        colors = np.array([sample_image(self.level_images[i], self.full_uv[i]+displacement(
            controls[i], self.full_bases[i])) for i in range(len(self.views))])
        visibility = self.full_visibility
        # A conditional mode proposal considers actual observed colors at each
        # point. It retains sharp alternatives without averaging chain hypotheses.
        candidate_cost = np.array([np.sum(visibility*residual_cost(colors, c[None],
                            self.sigma, self.options.outlier_rate), axis=0) for c in colors])
        candidate_cost[visibility < .001] = np.inf
        source = np.argmin(candidate_cost, axis=0)
        mode = colors[source, np.arange(len(current))]
        # An IRLS proposal handles ordinary small RGB noise around the current
        # mode. Large inconsistent colors receive low inlier responsibility.
        error = np.mean((colors-current[None])**2, axis=-1)
        inlier = (1-self.options.outlier_rate)*np.exp(-error/(2*self.sigma**2))
        uniform = self.options.outlier_rate*(2*np.pi*3*self.sigma**2)**1.5
        weights = visibility*inlier/(inlier+uniform)
        mean = np.sum(weights[..., None]*colors, axis=0)/np.maximum(weights.sum(axis=0)[:, None], 1e-8)
        # Both complete native-volume candidates are scored via the actual volume
        # interpolation. An attractive pointwise estimate can still be rejected.
        best, best_energy = old, before
        for surface in (mode, mean):
            candidate = self.volume_from_surface(surface)
            self.restore_texture(candidate)
            energy = self.energy(controls, observations)
            if energy < best_energy-1e-10:
                best, best_energy = candidate, energy
        self.restore_texture(best)
        self.texture_updates.append({'before': before, 'after': best_energy})
        return best is not old
