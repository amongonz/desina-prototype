"""Mesh-free signed-distance reconstruction. World axes: x right, y up, z front."""
from dataclasses import dataclass, field
import json
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from PIL import Image


@dataclass
class Camera:
    azimuth: float = 0
    elevation: float = 0
    scale: float = 2.6  # vertical AND horizontal span (square orthographic images)

    def __post_init__(self):
        if not np.isfinite([self.azimuth, self.elevation, self.scale]).all() or self.scale <= 0:
            raise ValueError('Camera values must be finite and scale must be positive.')

    def basis(self):
        a, e = np.radians([self.azimuth, self.elevation])
        front = np.array([np.sin(a)*np.cos(e), np.sin(e), np.cos(a)*np.cos(e)])
        right = np.array([np.cos(a), 0, -np.sin(a)])
        return right, np.cross(front, right), front

    def project(self, points, size):
        right, up, front = self.basis()
        u = (points @ right / self.scale + .5) * size - .5
        v = (.5 - points @ up / self.scale) * size - .5
        return u, v, points @ front

    def rays(self, size):
        right, up, front = self.basis()
        v, u = np.mgrid[:size, :size]
        return (((u.ravel()+.5)/size-.5)*self.scale)[:, None]*right + \
            ((.5-(v.ravel()+.5)/size)*self.scale)[:, None]*up + 3*front, -front


@dataclass
class Model:
    sdf: np.ndarray
    color: np.ndarray
    metadata: dict = field(default_factory=dict)

    @property
    def spacing(self):
        return 2/(self.sdf.shape[0]-1)

    def sample(self, points, volume=None, outside=1.):
        coords = ((np.asarray(points)+1)/self.spacing).T
        volume = self.sdf if volume is None else volume
        if volume.ndim == 4:
            return np.stack([ndi.map_coordinates(volume[..., c], coords, order=1,
                             mode='constant', cval=outside) for c in range(3)], axis=-1)
        return ndi.map_coordinates(volume, coords, order=1, mode='constant', cval=outside)

    def normals(self, points):
        eps = self.spacing
        grad = np.stack([self.sample(points+np.eye(3)[i]*eps) -
                         self.sample(points-np.eye(3)[i]*eps) for i in range(3)], -1)
        return grad / np.maximum(np.linalg.norm(grad, axis=-1, keepdims=True), 1e-8)

    def save(self, path):
        np.savez_compressed(path, sdf=self.sdf, color=self.color,
                            metadata=json.dumps(self.metadata))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            return cls(data['sdf'], data['color'], json.loads(str(data['metadata'])))


def grid(n):
    return np.stack(np.meshgrid(*([np.linspace(-1, 1, n)]*3), indexing='ij'), -1)


def redistance(inside, spacing):
    if not inside.any() or inside.all():
        raise ValueError('Shape is empty or fills the entire grid; check masks, cameras and scale.')
    # Half-voxel correction puts the boundary halfway between adjacent centers.
    return ((ndi.distance_transform_edt(~inside)-ndi.distance_transform_edt(inside)) *
            spacing - np.where(inside, -spacing/2, spacing/2)).astype(np.float32)


def trace(model, camera, size=192):
    """Conservative grid traversal, then sub-voxel bisection of first entry."""
    origins, direction = camera.rays(size)
    # The entire [-1,1]^3 box lies within radius sqrt(3).
    t = np.full(len(origins), 1.25)
    hit = np.zeros(len(t), bool)
    points = np.full_like(origins, np.nan)
    step = model.spacing * .65
    for _ in range(int(3.5/step)+2):
        ids = np.flatnonzero(~hit & (t < 4.75))
        if not len(ids):
            break
        p = origins[ids] + t[ids, None]*direction
        entered = model.sample(p) <= 0
        found = ids[entered]
        if len(found):
            lo, hi = t[found]-step, t[found].copy()
            for _ in range(6):
                mid = (lo+hi)/2
                negative = model.sample(origins[found]+mid[:, None]*direction) <= 0
                hi = np.where(negative, mid, hi)
                lo = np.where(negative, lo, mid)
            points[found] = origins[found]+hi[:, None]*direction
            hit[found] = True
        t[ids] += step
    return points.reshape(size, size, 3), hit.reshape(size, size)


def render(model, camera, size=192, shaded=True):
    points, mask = trace(model, camera, size)
    rgb = np.full((size, size, 3), .96)
    if mask.any():
        paint = model.sample(points[mask], model.color, outside=.65)
        if shaded:
            normals = model.normals(points[mask])
            light = np.array([-.4, .8, .65]); light /= np.linalg.norm(light)
            paint = paint * (.65 + .35*np.maximum(normals @ light, 0))[:, None]
        rgb[mask] = np.clip(paint, 0, 1)
    return rgb, mask, points


def save_image(path, rgb, mask=None):
    data = np.uint8(np.clip(rgb, 0, 1)*255+.5)
    if mask is not None:
        data = np.dstack([data, np.uint8(mask)*255])
    Image.fromarray(data).save(path)


@dataclass
class View:
    camera: Camera
    rgb: np.ndarray
    mask: np.ndarray
    name: str = ''


def load_project(path):
    path = Path(path)
    spec = json.loads(path.read_text())
    views = []
    for entry in spec['views']:
        image = np.asarray(Image.open(path.parent/entry['image']).convert('RGBA'))/255.
        if image.shape[0] != image.shape[1]:
            raise ValueError('Input images must be square; pad rather than stretch.')
        if 'mask' in entry:
            mask = np.asarray(Image.open(path.parent/entry['mask']).convert('L')) > 127
        elif np.any(image[..., 3] < 1):
            mask = image[..., 3] > .5
        else:
            # Closed outline / colored drawing on white. Open contours need an explicit mask.
            mask = ndi.binary_fill_holes(np.min(image[..., :3], axis=-1) < .92)
        if mask.shape != image.shape[:2] or not mask.any():
            raise ValueError('Each drawing needs a nonempty silhouette matching its dimensions.')
        camera = Camera(**entry.get('camera', {}))
        if not np.isfinite([camera.azimuth, camera.elevation, camera.scale]).all() or camera.scale <= 0:
            raise ValueError('Camera values must be finite and scale must be positive.')
        views.append(View(camera, image[..., :3], mask, entry['image']))
    if len(views) < 2:
        raise ValueError('Provide at least two registered views.')
    return views, spec


def reconstruct(views, resolution=80, tolerance=1., smooth=.7):
    if not 24 <= resolution <= 192:
        raise ValueError('Resolution must be between 24 and 192.')
    if not np.isfinite([tolerance, smooth]).all() or tolerance < 0 or smooth < 0:
        raise ValueError('Tolerance and smoothing must be finite and nonnegative.')
    xyz = grid(resolution).reshape(-1, 3)
    field = np.full(len(xyz), -np.inf)
    for view in views:
        n = len(view.mask)
        signed = ndi.distance_transform_edt(~view.mask)-ndi.distance_transform_edt(view.mask)
        u, v, _ = view.camera.project(xyz, n)
        constraint = ndi.map_coordinates(signed, [v, u], order=1, mode='constant', cval=n)
        field = np.maximum(field, (constraint-tolerance)*view.camera.scale/n)
    inside = field.reshape((resolution,)*3) <= 0
    sdf = redistance(inside, 2/(resolution-1))
    if smooth:
        sdf = ndi.gaussian_filter(sdf, smooth)
    model = Model(sdf, np.full((*sdf.shape, 3), .65, np.float32),
                  {'resolution': resolution, 'tolerance_pixels': tolerance,
                   'smooth_voxels': smooth, 'views': [v.name for v in views], 'edits': []})
    bake_paint(model, views)
    return model


def bake_paint(model, views):
    """Fuse only visible, front-facing samples; extend missing color by nearest surface."""
    shell = np.abs(model.sdf) < model.spacing*2
    indices = np.argwhere(shell)
    p = indices*model.spacing-1
    normals = model.normals(p)
    total = np.zeros(len(p))
    colors = np.zeros((len(p), 3))
    for view in views:
        n = len(view.mask)
        u, v, depth = view.camera.project(p, n)
        hits, mask = trace(model, view.camera, n)
        front = view.camera.basis()[2]
        z = np.full((n, n), -10.)
        z[mask] = hits[mask] @ front
        visible_depth = ndi.map_coordinates(z, [v, u], order=0, mode='constant', cval=-10)
        valid = ndi.map_coordinates(view.mask.astype(float), [v, u], order=0,
                                    mode='constant', cval=0)
        weight = np.maximum(normals @ front, 0)**4 * valid * \
            (np.abs(visible_depth-depth) < model.spacing*3)
        # Extend edge paint outside the mask so bilinear sampling cannot bleed white.
        nearest = ndi.distance_transform_edt(~view.mask, return_distances=False, return_indices=True)
        extended = view.rgb[tuple(nearest)]
        color = np.stack([ndi.map_coordinates(extended[..., c], [v, u], order=1,
                                             mode='nearest') for c in range(3)], -1)
        colors += color*weight[:, None]
        total += weight
    known = total > 1e-6
    if not known.any():
        raise ValueError('No visible paint samples; check camera registration.')
    colors[known] /= total[known, None]
    # A spatial color field, never queried by viewing direction. Unseen areas are guesses.
    tree = cKDTree(p[known])
    _, nearest = tree.query(p[~known])
    colors[~known] = colors[known][nearest]
    model.metadata['paint_coverage'] = float(known.mean())
    # Extend shell colors to the volume for robust sculpting and interpolation.
    nearest = ndi.distance_transform_edt(~shell, return_distances=False, return_indices=True)
    model.color[tuple(indices.T)] = colors
    model.color = model.color[tuple(nearest)].astype(np.float32)


def brush(model, center, radius=.12, mode='paint', color=(.9, .25, .1), strength=1.):
    center = np.asarray(center, float)
    if center.shape != (3,) or not np.isfinite(center).all() or not 0 < radius <= 2:
        raise ValueError('Brush needs a finite 3D center and radius in (0,2].')
    if not 0 < strength <= 1 or mode not in ('paint', 'add', 'subtract', 'smooth'):
        raise ValueError('Invalid brush mode or strength (0,1].')
    color = np.asarray(color, float)
    if color.shape != (3,) or not np.isfinite(color).all() or np.any((color < 0) | (color > 1)):
        raise ValueError('Paint color needs three values in [0,1].')
    distance = np.linalg.norm(grid(len(model.sdf))-center, axis=-1)
    weight = np.clip(1-distance/radius, 0, 1)*strength
    if mode == 'paint':
        model.color += weight[..., None]*(color-model.color)
    elif mode == 'smooth':
        model.sdf += weight*(ndi.gaussian_filter(model.sdf, 1.5)-model.sdf)
    else:
        sphere = distance-radius
        target = np.minimum(model.sdf, sphere) if mode == 'add' else np.maximum(model.sdf, -sphere)
        model.sdf += strength*(target-model.sdf)
    model.metadata.setdefault('edits', []).append(dict(center=center.tolist(), radius=radius,
                                                      mode=mode, color=color.tolist(), strength=strength))
