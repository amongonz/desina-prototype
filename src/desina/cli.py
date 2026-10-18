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
    paint_demo = commands.add_parser('paint-demo', help='Compare Monte Carlo paint fitting on inconsistent artwork')
    paint_demo.add_argument('--out', type=Path, default=Path('examples/paint-shared'))
    paint_demo.add_argument('--resolution', type=int, default=80)
    paint_demo.add_argument('--size', type=int, default=160)
    paint_demo.add_argument('--distortion', type=float, default=.035, help='Input paint distortion as image fraction')
    paint_demo.add_argument('--no-turntable', action='store_true')
    benchmark = commands.add_parser('paint-benchmark', help='Run the predeclared fresh corruption suite')
    benchmark.add_argument('--out', type=Path, default=Path('examples/paint-validation'))
    paint_fit = commands.add_parser('fit-paint', help='Fit inconsistent painted views to an existing volume')
    paint_fit.add_argument('model', type=Path)
    paint_fit.add_argument('project', type=Path)
    paint_fit.add_argument('--out', type=Path, required=True)
    for command in (paint_demo, paint_fit):
        command.add_argument('--seed', type=int, default=7)
        command.add_argument('--steps', type=int, default=350, help='Proposals per view per scale per chain')
        command.add_argument('--chains', type=int, default=3)
        command.add_argument('--samples', type=int, default=1800, help='Random surface quadrature count')
        command.add_argument('--knots', type=int, default=4)
        command.add_argument('--max-warp', type=float, default=.065, help='Maximum displacement as image fraction')
        command.add_argument('--reference', type=int, default=-1, help='Pin a view index; -1 uses geometric priors only')
        command.add_argument('--texture-resolution', type=int, default=128)
        command.add_argument('--displacement-prior', type=float, default=.025)
        command.add_argument('--smoothness-prior', type=float, default=.06)
        command.add_argument('--outlier-rate', type=float, default=.12)
        command.add_argument('--posterior-draws', type=int, default=12)
        command.add_argument('--temperature', type=float, default=.008)
        command.add_argument('--objective', choices=['shared', 'pairwise'], default='shared')
        command.add_argument('--texture-every', type=int, default=50)
        command.add_argument('--texture-prior', type=float, default=.025)
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
    view.add_argument('--uncertainty', action='store_true', help='Render saved paint-fit uncertainty instead of RGB')
    view.add_argument('--hypothesis', type=int, help='Render a retained chain texture by index')
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
        elif args.command == 'paint-benchmark':
            from .paint_benchmark import run_benchmark
            run_benchmark(args.out)
        elif args.command in ('fit-paint', 'paint-demo'):
            from .paint import PaintOptions, fit_paint
            options = PaintOptions(seed=args.seed, steps=args.steps, chains=args.chains,
                samples=args.samples, knots=args.knots, max_warp=args.max_warp, reference=args.reference)
            options.texture_resolution = args.texture_resolution
            options.displacement_prior = args.displacement_prior
            options.smoothness_prior = args.smoothness_prior
            options.outlier_rate = args.outlier_rate
            options.posterior_draws = args.posterior_draws
            options.temperature = args.temperature
            options.objective = args.objective
            options.texture_every = args.texture_every
            options.texture_prior = args.texture_prior
            if args.command == 'paint-demo':
                from .paint_demo import run_paint_demo
                run_paint_demo(args.out, args.resolution, args.size, options,
                               args.distortion, not args.no_turntable)
            else:
                model = Model.load(args.model)
                views, _ = load_project(args.project)
                report = fit_paint(model, views, options, progress=lambda s: print(s, flush=True))
                model.save(args.out)
                print(json.dumps(report, indent=2))
        elif args.command == 'reconstruct':
            views, _ = load_project(args.project)
            model = reconstruct(views, args.resolution, args.tolerance, args.smooth)
            model.save(args.out)
            print(json.dumps(model.metadata, indent=2))
        elif args.command == 'render':
            if not 16 <= args.size <= 1024 or args.scale <= 0:
                raise ValueError('Size must be 16..1024 and scale must be positive.')
            model = Model.load(args.model)
            if args.hypothesis is not None:
                if args.uncertainty:
                    raise ValueError('Uncertainty describes the complete fit; render it without --hypothesis.')
                if model.paint_hypothesis_colors is None or not 0 <= args.hypothesis < len(model.paint_hypothesis_colors):
                    raise ValueError('This model has no texture hypothesis with that index.')
                model.color = model.paint_hypothesis_colors[args.hypothesis]
            rgb, mask, points = render(model, Camera(args.azimuth, args.elevation, args.scale),
                                       args.size, not args.flat)
            if args.uncertainty:
                if model.uncertainty is None:
                    raise ValueError('This model has no saved paint-fit uncertainty.')
                from .paint_demo import uncertainty_color
                rgb[mask] = uncertainty_color(model.sample(points[mask], model.uncertainty))
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
