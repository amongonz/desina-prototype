"""Feature-free Monte Carlo registration of inconsistent surface paintings.

Only NumPy/SciPy, image samples, an existing SDF, and camera hints are used.
Warp coordinates are fractions of each input image, never viewing-dependent paint.
"""
from dataclasses import asdict, dataclass
import time

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

from .core import Model, trace


@dataclass
class PaintOptions:
    seed: int = 7
    steps: int = 350  # proposals per movable view per pyramid level per chain
    chains: int = 3
    samples: int = 1800  # surface quadrature samples, not detected image features
    knots: int = 4
    max_warp: float = .065  # fraction of image width
    displacement_prior: float = .025
    smoothness_prior: float = .06
    outlier_rate: float = .12
    reference: int = -1  # -1: geometry-centered displacement prior; otherwise pin one view
    posterior_draws: int = 12
    texture_resolution: int = 128
    temperature: float = .008
    final_temperature: float = .00008
    tail_temperature: float = .00015
    objective: str = "shared"
    texture_every: int = 50
    texture_prior: float = .025

    def validate(self, nviews):
        if self.objective not in ("shared", "pairwise") or self.texture_every < 1 or not np.isfinite(self.texture_prior) or self.texture_prior < 0:
            raise ValueError("Invalid texture objective, update interval, or prior.")
        if not (2 <= nviews and -1 <= self.reference < nviews):
            raise ValueError('Need at least two views and a valid reference index.')
        if not (1 <= self.steps <= 20000 and 1 <= self.chains <= 16 and
                100 <= self.samples <= 20000 and 2 <= self.knots <= 8 and
                1 <= self.posterior_draws <= 128 and 24 <= self.texture_resolution <= 192):
            raise ValueError('Invalid sampling budget, chain count, or control-grid size.')
        values = [self.max_warp, self.displacement_prior, self.smoothness_prior, self.outlier_rate,
                  self.temperature, self.final_temperature, self.tail_temperature]
        if not np.isfinite(values).all() or not (0 < self.max_warp <= .15 and
                self.displacement_prior > 0 and self.smoothness_prior > 0 and
                0 < self.outlier_rate < .5 and min(values[-3:]) > 0):
            raise ValueError('Warp/prior scales must be positive; outlier rate must be in (0,.5).')


def warp_basis(uv, knots):
    """Four bilinear weights per point; uv and displacement use image fractions."""
    xy = np.clip(uv, 0, 1)*(knots-1)
    lo = np.minimum(xy.astype(int), knots-2)
    f = xy-lo
    u, v = lo.T
    a, b = f.T
    index = np.stack([v*knots+u, v*knots+u+1, (v+1)*knots+u, (v+1)*knots+u+1], -1)
    weight = np.stack([(1-a)*(1-b), a*(1-b), (1-a)*b, a*b], -1)
    return index, weight


def displacement(controls, basis):
    index, weight = basis
    return np.sum(controls.reshape(-1, 2)[index]*weight[..., None], axis=1)


def regularization(controls, options):
    """Bounds plus a contraction bound: I + displacement cannot locally fold."""
    if np.max(np.abs(controls)) > options.max_warp:
        return np.inf
    du = np.diff(controls, axis=1)*(options.knots-1)
    dv = np.diff(controls, axis=0)*(options.knots-1)
    # Bound every cell-corner Jacobian's Frobenius norm below one.
    for x in [du[:-1], du[1:]]:
        for y in [dv[:, :-1], dv[:, 1:]]:
            if np.max(np.sum(x*x+y*y, axis=-1)) >= .8**2:
                return np.inf
    return (options.displacement_prior*np.mean(controls**2) +
            options.smoothness_prior*(np.mean(np.diff(controls, axis=0)**2)+
                                      np.mean(np.diff(controls, axis=1)**2))) / options.max_warp**2


def propose(controls, rng, scale):
    """Small editable proposal family: translation, affine tilt, or Gaussian bump."""
    g = len(controls)
    v, u = np.mgrid[0:1:complex(g), 0:1:complex(g)]
    kind = rng.integers(3)
    delta = rng.normal(0, scale, 2)
    if kind == 0:
        field = np.ones((g, g))
    elif kind == 1:
        angle = rng.uniform(0, 2*np.pi)
        field = (u-.5)*np.cos(angle)+(v-.5)*np.sin(angle)
    else:
        cx, cy = rng.uniform(.1, .9, 2)
        width = rng.uniform(.18, .45)
        field = np.exp(-((u-cx)**2+(v-cy)**2)/(2*width**2))
    # Bounds are enforced by rejecting proposals, not clipping (which biases moves).
    return controls+field[..., None]*delta


def prepare_surface(model, resolution):
    # Adjacent shell voxels provide integration samples; project centers onto the zero set.
    sdf = ndi.zoom(model.sdf, resolution/len(model.sdf), order=1)
    spacing = 2/(resolution-1)
    shell = np.abs(sdf) < spacing*.8
    indices = np.argwhere(shell)
    if not len(indices):
        raise ValueError('The model has no sampled zero-level surface.')
    points = indices*spacing-1
    normals = model.normals(points)
    points -= model.sample(points)[:, None]*normals
    normals = model.normals(points)
    return shell, indices, points, normals


def prepare_views(model, views, points, normals):
    uv, visibility, extended = [], [], []
    for view in views:
        n = len(view.mask)
        u, v, depth = view.camera.project(points, n)
        uv.append(np.stack([(u+.5)/n, (v+.5)/n], -1))
        hits, hit = trace(model, view.camera, n)
        front = view.camera.basis()[2]
        z = np.full((n, n), -10.)
        z[hit] = hits[hit] @ front
        nearest_z = ndi.map_coordinates(z, [v, u], order=0, mode='constant', cval=-10)
        valid = ndi.map_coordinates(view.mask.astype(float), [v, u], order=0,
                                    mode='constant', cval=0)
        visibility.append(np.maximum(normals @ front, 0)**2 * valid *
                          (np.abs(depth-nearest_z) < model.spacing*2.5))
        nearest = ndi.distance_transform_edt(~view.mask, return_distances=False, return_indices=True)
        extended.append(view.rgb[tuple(nearest)])
    return np.array(uv), np.array(visibility), extended


def sample_image(image, uv):
    n = len(image)
    coordinates = [uv[:, 1]*n-.5, uv[:, 0]*n-.5]
    return np.stack([ndi.map_coordinates(image[..., c], coordinates, order=1, mode='nearest')
                     for c in range(3)], -1)


def residual_cost(a, b, sigma, outlier_rate):
    """Fixed inlier/uniform-outlier mixture. Rejection never has zero cost."""
    error = np.mean((a-b)**2, axis=-1)
    # The channel standard deviation is sqrt(3)*sigma because error is RGB MSE.
    # Drop the Gaussian normalization (constant during a stage) from the log loss.
    gaussian_volume = (2*np.pi*3*sigma*sigma)**1.5
    return -np.log((1-outlier_rate)*np.exp(-error/(2*sigma*sigma)) +
                   outlier_rate*gaussian_volume)


class Registration:
    def __init__(self, views, uv, visibility, images, options):
        self.views, self.uv, self.visibility, self.images, self.options = views, uv, visibility, images, options
        self.bases = [warp_basis(x, options.knots) for x in uv]
        self.pair_weight = np.sqrt(visibility[:, None]*visibility[None, :])
        for i in range(len(views)):
            self.pair_weight[i, i] = 0
        self.normalizer = self.pair_weight.sum()/2
        if self.normalizer <= 0:
            raise ValueError('Views need overlapping visible surface to fit paint correspondence.')
        self.level_images = images
        self.sigma = .1

    def observations(self, controls):
        return np.array([self.observe(i, controls[i]) for i in range(len(self.views))])

    def observe(self, i, controls):
        return sample_image(self.level_images[i], self.uv[i]+displacement(controls, self.bases[i]))

    def prior_loss(self, i, controls):
        shifted = self.uv[i]+displacement(controls, self.bases[i])
        n = len(self.views[i].mask)
        support = ndi.map_coordinates(self.views[i].mask.astype(float),
                    [shifted[:, 1]*n-.5, shifted[:, 0]*n-.5], order=1, mode='constant', cval=0)
        missing = np.sum((1-support)*self.visibility[i])/max(self.visibility[i].sum(), 1e-8)
        return regularization(controls, self.options)+missing*.5

    def local_loss(self, i, controls, observation, all_observations):
        costs = residual_cost(observation[None], all_observations, self.sigma, self.options.outlier_rate)
        data = np.sum(costs*self.pair_weight[i])*len(self.views)/self.normalizer
        return data+self.prior_loss(i, controls)

    def texture_state(self):
        return None

    def restore_texture(self, state):
        pass

    def refine_texture(self, controls, observations):
        return False

    def energy(self, controls, observations):
        # local losses each count their pairwise term twice; priors are counted once.
        data = 0.
        for i in range(len(self.views)):
            for j in range(i):
                data += np.sum(self.pair_weight[i, j]*residual_cost(
                    observations[i], observations[j], self.sigma, self.options.outlier_rate))
        priors = []
        for i in range(len(self.views)):
            costs = residual_cost(observations[i][None], observations, self.sigma, self.options.outlier_rate)
            local_data = np.sum(costs*self.pair_weight[i])*len(self.views)/self.normalizer
            priors.append(self.local_loss(i, controls[i], observations[i], observations)-local_data)
        return float(data/self.normalizer+np.mean(priors))


def register(reg, options, progress=None):
    """Annealed conditional MH with monotone canonical-texture proposals.

    Warp tails condition on each chain's texture. Chain winners remain separate
    alternatives; neither this short run nor the texture updates imply convergence.
    """
    rng = np.random.default_rng(options.seed)
    nviews = len(reg.views)
    shape = (nviews, options.knots, options.knots, 2)
    movable = [i for i in range(nviews) if i != options.reference]
    initial_texture = reg.texture_state()
    winners, chain_logs, retained = [], [], []
    for chain in range(options.chains):
        reg.restore_texture(initial_texture)
        controls = np.zeros(shape)
        if chain:
            controls = ndi.gaussian_filter(rng.normal(0, options.max_warp*.12, shape),
                                           sigma=(0, .8, .8, 0))
            if options.reference >= 0:
                controls[options.reference] = 0
        accepted = attempted = 0
        history = []
        for stage, (blur, sigma) in enumerate([(2.2, .16), (1., .10), (0., .065)]):
            # Blur scales track image resolution, with 160px as the original setting.
            reg.level_images = [ndi.gaussian_filter(img, (blur*len(img)/160,)*2+(0,))
                                if blur else img for img in reg.images]
            reg.sigma = sigma
            obs = reg.observations(controls)
            reg.refine_texture(controls, obs)
            current = reg.energy(controls, obs)
            best, best_energy, best_texture = controls.copy(), current, reg.texture_state()
            for sweep in range(options.steps):
                fraction = sweep/max(options.steps-1, 1)
                temperature = options.temperature*(1-fraction)**2+options.final_temperature
                proposal_scale = options.max_warp*(.20*(1-fraction)+.015)/(1+stage*.35)
                for i in rng.permutation(movable):
                    candidate = propose(controls[i], rng, proposal_scale)
                    attempted += 1
                    if not np.isfinite(regularization(candidate, options)):
                        continue
                    color = reg.observe(i, candidate)
                    delta = (reg.local_loss(i, candidate, color, obs)-
                             reg.local_loss(i, controls[i], obs[i], obs))/nviews
                    if delta <= 0 or np.log(rng.random()) < -delta/temperature:
                        controls[i], obs[i] = candidate, color
                        current += delta
                        accepted += 1
                        # Record EVERY accepted state, not every tenth sweep.
                        if current < best_energy:
                            best, best_energy, best_texture = controls.copy(), current, reg.texture_state()
                if (sweep+1) % options.texture_every == 0:
                    reg.refine_texture(controls, obs)
                    current = reg.energy(controls, obs)
                    if current < best_energy:
                        best, best_energy, best_texture = controls.copy(), current, reg.texture_state()
            controls = best
            reg.restore_texture(best_texture)
            best_energy = reg.energy(controls, reg.observations(controls))
            history.append(best_energy)
            if progress:
                progress(f'chain {chain+1}/{options.chains}, scale {stage+1}/3: energy {best_energy:.4f}')
        pre_tail = best_energy
        # Each chain keeps its own fixed-texture tail instead of discarding all
        # alternatives to the first winning chain. Export also tracks tail improvements.
        obs = reg.observations(controls)
        current = reg.energy(controls, obs)
        draws = []
        for step in range(options.posterior_draws*8):
            for i in movable:
                candidate = propose(controls[i], rng, options.max_warp*.025)
                if not np.isfinite(regularization(candidate, options)):
                    continue
                color = reg.observe(i, candidate)
                delta = (reg.local_loss(i, candidate, color, obs)-
                         reg.local_loss(i, controls[i], obs[i], obs))/nviews
                if delta <= 0 or np.log(rng.random()) < -delta/options.tail_temperature:
                    controls[i], obs[i] = candidate, color
                    current += delta
                    if current < best_energy:
                        best, best_energy, best_texture = controls.copy(), current, reg.texture_state()
            if step % 8 == 7:
                draws.append(controls.copy())
        reg.restore_texture(best_texture)
        best_energy = reg.energy(best, reg.observations(best))
        winners.append((best_energy, best.copy(), best_texture))
        retained.extend(draws)
        chain_logs.append({'chain': chain, 'energies': history, 'pre_tail_energy': pre_tail,
                           'best_energy': best_energy, 'acceptance_rate': accepted/max(attempted, 1)})
    # Identity gets a conditional texture fit too, providing a fair fallback.
    reg.restore_texture(initial_texture)
    identity = np.zeros(shape)
    identity_obs = reg.observations(identity)
    reg.refine_texture(identity, identity_obs)
    fallback = (reg.energy(identity, identity_obs), identity, reg.texture_state())
    winner = min(winners+[fallback], key=lambda x: x[0])
    energy, best, texture = winner
    reg.restore_texture(texture)
    hypothesis_chains = sorted(range(len(winners)), key=lambda i: winners[i][0])
    reg.hypotheses = [winners[i] for i in hypothesis_chains]
    if winner is fallback:
        reg.hypotheses.insert(0, fallback)
        hypothesis_chains.insert(0, -1)
    return best, np.array(retained), {'map_energy': energy, 'chains': chain_logs,
        'tail_temperature': options.tail_temperature, 'objective': options.objective,
        'sample_chain_indices': np.repeat(np.arange(options.chains), options.posterior_draws).tolist(),
        'hypothesis_chain_indices': hypothesis_chains,
        'hypothesis_energies': [h[0] for h in reg.hypotheses]}


def source_graph(points):
    """One undirected Potts graph; greedy coloring permits safe parallel updates."""
    count = len(points)
    if count < 2:
        return np.empty((0, 2), int), [np.array([0])], np.zeros((1, 1), int), np.zeros((1, 1))
    neighbor = cKDTree(points).query(points, k=min(7, count))[1][:, 1:]
    edges = np.stack([np.repeat(np.arange(count), neighbor.shape[1]), neighbor.ravel()], -1)
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    edges = edges[edges[:, 0] != edges[:, 1]]
    adjacency = [[] for _ in range(count)]
    for a, b in edges:
        adjacency[a].append(b)
        adjacency[b].append(a)
    degree = max(map(len, adjacency))
    padded, valid = np.zeros((count, degree), int), np.zeros((count, degree))
    coloring = np.full(count, -1, int)
    for i, neighbors in enumerate(adjacency):
        padded[i, :len(neighbors)], valid[i, :len(neighbors)] = neighbors, 1
        used = set(coloring[neighbors])
        color = 0
        while color in used:
            color += 1
        coloring[i] = color
    groups = [np.flatnonzero(coloring == color) for color in range(coloring.max()+1)]
    # A single edge weight defines the same energy during scoring and updates.
    valid *= .24/(2*len(edges)/count)
    return edges, groups, padded, valid


def smooth_sources(unary, points, sweeps=4):
    """Block-coordinate ICM; every block is an independent set, so energy decreases."""
    count = len(points)
    source = np.argmin(unary, axis=0)
    edges, groups, neighbors, weight = source_graph(points)
    edge_weight = float(weight.max())
    def energy():
        return float(unary[source, np.arange(count)].sum()+
                     edge_weight*np.sum(source[edges[:, 0]] != source[edges[:, 1]]))
    history = [energy()]
    for _ in range(sweeps):
        changed = False
        for group in groups:
            costs = unary[:, group].copy()
            for i in range(len(unary)):
                costs[i] += np.sum(weight[group]*(source[neighbors[group]] != i), axis=1)
            proposed = np.argmin(costs, axis=0)
            improve = costs[proposed, np.arange(len(group))] < costs[source[group], np.arange(len(group))]-1e-12
            changed |= bool(improve.any())
            source[group[improve]] = proposed[improve]
        history.append(energy())
        if not changed:
            break
    return source, history


def fuse(points, colors, visibility, options):
    """A robust color mode, spatially coherent source support, never a mode average."""
    nviews, count, _ = colors.shape
    agreement = np.zeros((nviews, count))
    weight = visibility/np.maximum(visibility.sum(axis=0), 1e-8)
    for i in range(nviews):
        agreement[i] = np.sum(weight*residual_cost(colors[i][None], colors, .075,
                                                 options.outlier_rate), axis=0)
    unary = agreement-.20*np.log(np.maximum(visibility, 1e-6))
    unary[visibility < .001] = 1e4
    source, _ = smooth_sources(unary, points)
    selected = colors[source, np.arange(count)]
    supported = np.any(visibility >= .001, axis=0)
    if not supported.any():
        raise ValueError('No visible input paint samples.')
    if not supported.all():
        nearest = cKDTree(points[supported]).query(points[~supported])[1]
        selected[~supported] = selected[supported][nearest]
    disagreement = np.sum(weight*np.mean((colors-selected[None])**2, axis=-1), axis=0)
    # Include low coverage; one view cannot establish cross-view agreement.
    effective_views = (visibility > .03).sum(axis=0)
    uncertainty = np.clip(np.sqrt(disagreement)/.20 + .25*(effective_views < 2), 0, 1)
    uncertainty[~supported] = 1
    return selected, uncertainty, source


def fit_paint(model, views, options=None, progress=None, align=True):
    options = options or PaintOptions()
    options.validate(len(views))
    start = time.perf_counter()
    shell, indices, points, normals = prepare_surface(model, options.texture_resolution)
    uv, visibility, images = prepare_views(model, views, points, normals)
    rng = np.random.default_rng(options.seed)
    eligible = np.flatnonzero((visibility > .015).sum(axis=0) >= 2)
    if not len(eligible):
        raise ValueError('No jointly visible paint samples to register.')
    chosen = rng.choice(eligible, min(len(eligible), options.samples), replace=False)
    if options.objective == 'shared':
        from .paint_joint import SharedTextureRegistration
        reg = SharedTextureRegistration(model, views, uv[:, chosen], visibility[:, chosen], images,
            options, points[chosen], uv, visibility, points, indices, shell)
    else:
        reg = Registration(views, uv[:, chosen], visibility[:, chosen], images, options)
    reg.level_images = images
    reg.sigma = .065
    identity = np.zeros((len(views), options.knots, options.knots, 2))
    initial_energy = reg.energy(identity, reg.observations(identity))
    if align:
        controls, draws, diagnostics = register(reg, options, progress)
    else:
        controls, draws = identity, identity[None]
        reg.refine_texture(identity, reg.observations(identity))
        diagnostics = {'map_energy': reg.energy(identity, reg.observations(identity)), 'chains': [],
                       'objective': options.objective, 'sample_chain_indices': [0]}
        reg.hypotheses = [(diagnostics['map_energy'], identity, reg.texture_state())]
    colors = np.array([sample_image(images[i], uv[i]+displacement(
        controls[i], warp_basis(uv[i], options.knots))) for i in range(len(views))])
    if options.objective == 'shared':
        # This is the selected joint state, not a post-hoc fusion of its images.
        selected = model.sample(points, reg.texture_state())
        weight = visibility/np.maximum(visibility.sum(axis=0), 1e-8)
        disagreement = np.sum(weight*np.mean((colors-selected[None])**2, axis=-1), axis=0)
        uncertainty = np.clip(np.sqrt(disagreement)/.20+.25*((visibility > .03).sum(axis=0) < 2), 0, 1)
        uncertainty[~reg.supported] = 1
        source = np.argmax(visibility, axis=0)  # coverage diagnostic, not an output-source assignment
    else:
        selected, uncertainty, source = fuse(points, colors, visibility, options)
    # Preserve up to three complete chain interpretations, rather than conflating
    # different textures with uncertainty around a single conditional warp tail.
    hypotheses = reg.hypotheses[:3]
    model.paint_hypothesis_warps = np.array([h[1] for h in hypotheses], np.float32)
    model.paint_hypothesis_energies = np.array([h[0] for h in hypotheses])
    if options.objective == 'shared':
        model.paint_hypothesis_colors = np.array([h[2] for h in hypotheses], np.float32)
        alternatives = np.array([model.sample(points, h[2]) for h in hypotheses])
        uncertainty = np.clip(uncertainty+np.sqrt(np.var(alternatives, axis=0).mean(axis=-1))/.2, 0, 1)
    else:
        model.paint_hypothesis_colors = None
    spread = np.zeros(len(points))
    per_view_spread = []
    for i in range(len(views)):
        basis = warp_basis(uv[i], options.knots)
        shifts = np.array([displacement(draw[i], basis) for draw in draws])
        std = np.sqrt(np.var(shifts, axis=0).sum(axis=-1))
        per_view_spread.append(float(np.mean(std)*len(views[i].mask)))
        spread += std*visibility[i]
    spread /= np.maximum(visibility.sum(axis=0), 1e-8)*options.max_warp
    uncertainty = np.clip(uncertainty+spread, 0, 1)
    # Scatter one selected mode to the full volume. Never average warp hypotheses.
    nearest = ndi.distance_transform_edt(~shell, return_distances=False, return_indices=True)
    volume = np.zeros((*shell.shape, 3), np.float32)
    volume[tuple(indices.T)] = selected
    model.color = (reg.texture_state() if options.objective == 'shared' else
                   volume[tuple(nearest)].astype(np.float32))
    unc = np.ones(shell.shape, np.float32)
    unc[tuple(indices.T)] = uncertainty
    model.uncertainty = unc[tuple(nearest)].astype(np.float32)
    model.paint_warps = controls.astype(np.float32)
    model.paint_samples = draws.astype(np.float32)
    diagnostics.update({'initial_energy': initial_energy, 'seconds': time.perf_counter()-start,
                        'options': asdict(options), 'aligned': align,
                        'quadrature_samples': len(chosen),
                        'visible_shell_fraction': float((visibility.sum(axis=0) > .001).mean()),
                        'mean_uncertainty': float(uncertainty.mean()),
                        'source_fraction': [float(np.mean(source == i)) for i in range(len(views))],
                        'view_names': [v.name for v in views]})
    diagnostics['tail_displacement_std_pixels'] = per_view_spread
    if options.objective == 'shared':
        diagnostics['texture_updates'] = reg.texture_updates
        diagnostics['export_energy'] = reg.energy(controls, reg.observations(controls))
    diagnostics['hypothesis_count'] = len(hypotheses)
    model.metadata['paint_fit'] = diagnostics
    return diagnostics
