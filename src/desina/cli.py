import argparse
import json
from pathlib import Path
from .core import Camera, Model, brush, load_project, reconstruct, render, save_image


def main():
    parser = argparse.ArgumentParser(description='Desina — drawings to editable, painted implicit 3D')
    commands = parser.add_subparsers(dest='command', required=True)
    demo = commands.add_parser('demo', help='Generate synthetic inputs and evaluate unseen cameras')
    demo.add_argument('--out', type=Path, default=Path('examples/bird'))
    demo.add_argument('--resolution', type=int, default=80)
    demo.add_argument('--size', type=int, default=160)
    fit = commands.add_parser('reconstruct', help='Fit a volume and paint from a project JSON')
    fit.add_argument('project', type=Path)
    fit.add_argument('--out', type=Path, default=Path('model.npz'))
    fit.add_argument('--resolution', type=int, default=80)
    fit.add_argument('--tolerance', type=float, default=1., help='Silhouette slack in input pixels')
    fit.add_argument('--smooth', type=float, default=.7, help='Gaussian smoothing in voxels')
    view = commands.add_parser('render', help='Render a model from an arbitrary orthographic camera')
    view.add_argument('model', type=Path)
    view.add_argument('--out', type=Path, default=Path('render.png'))
    view.add_argument('--azimuth', type=float, default=30)
    view.add_argument('--elevation', type=float, default=20)
    view.add_argument('--scale', type=float, default=2.6)
    view.add_argument('--size', type=int, default=256)
    view.add_argument('--flat', action='store_true')
    edit = commands.add_parser('edit', help='Apply a JSON list of 3D paint/sculpt strokes')
    edit.add_argument('model', type=Path)
    edit.add_argument('strokes', type=Path)
    edit.add_argument('--out', type=Path, required=True)
    gui = commands.add_parser('studio', help='Open desktop orbit / paint / sculpt editor (requires Tk)')
    gui.add_argument('model', type=Path)
    args = parser.parse_args()
    try:
        if args.command == 'demo':
            from .demo import run_demo
            run_demo(args.out, args.resolution, args.size)
        elif args.command == 'reconstruct':
            views, _ = load_project(args.project)
            model = reconstruct(views, args.resolution, args.tolerance, args.smooth)
            model.save(args.out)
            print(json.dumps(model.metadata, indent=2))
        elif args.command == 'render':
            if not 16 <= args.size <= 1024 or args.scale <= 0:
                raise ValueError('Size must be 16..1024 and scale must be positive.')
            rgb, _, _ = render(Model.load(args.model), Camera(args.azimuth, args.elevation, args.scale),
                               args.size, not args.flat)
            save_image(args.out, rgb)
        elif args.command == 'edit':
            model = Model.load(args.model)
            for stroke in json.loads(args.strokes.read_text()):
                brush(model, **stroke)
            model.save(args.out)
        elif args.command == 'studio':
            from .studio import launch
            launch(args.model)
    except (ValueError, OSError, KeyError) as error:
        parser.exit(2, f'desina: {error}\n')
