"""Command-line entry point; --help does not load PyTorch or model assets."""

import argparse
from pathlib import Path
import sys
from .bundle import Bundle, validate_request, validate_sampling


def build_parser():
    parser = argparse.ArgumentParser(description="AMBIT spatial audio generation and editing")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("generate", "edit"):
        sub = commands.add_parser(name, help=f"{name.capitalize()} native FOA audio")
        sub.add_argument(
            "--bundle", required=True, type=Path, help="Local inference bundle directory"
        )
        sub.add_argument("--prompt", required=True, help="Scene description or edit instruction")
        if name == "edit":
            sub.add_argument("--source", required=True, type=Path, help="44.1 kHz W Y Z X FOA file")
        sub.add_argument(
            "--output",
            required=True,
            type=Path,
            help="New .wav file; also writes a .json ScenePlan",
        )
        sub.add_argument("--device", default="cuda", help="cpu, cuda, or cuda:N")
        sub.add_argument("--seed", type=int, default=42)
        sub.add_argument("--steps", type=int, help="Override the bundle's Euler step count")
        sub.add_argument("--cfg-scale", type=float, help="Override the bundle's guidance scale")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        validate_request(args.prompt, args.seed)
        task = "generation" if args.command == "generate" else "editing"
        bundle = Bundle.open(args.bundle, task=task)
        steps = bundle.steps if args.steps is None else args.steps
        cfg = bundle.cfg_scale if args.cfg_scale is None else args.cfg_scale
        validate_sampling(steps, cfg)
        from .audio import output_paths, read_foa

        output_paths(args.output)
        source = read_foa(args.source) if task == "editing" else None
        from .pipeline import GenerationPipeline, EditingPipeline

        cls = GenerationPipeline if task == "generation" else EditingPipeline
        pipeline = cls.from_bundle(bundle, device=args.device)
        options = dict(seed=args.seed, steps=steps, cfg_scale=cfg)
        result = (
            pipeline.generate(args.prompt, **options)
            if source is None
            else pipeline.edit(source, args.prompt, **options)
        )
        (wav, plan) = result.save(args.output)
        print(f"Audio: {wav}\nScenePlan: {plan}")
        return 0
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, ImportError) as exc:
        print(f"AMBIT: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
