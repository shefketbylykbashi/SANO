"""Top-level orchestrator.

Usage:
    python -m sano_py                  # full pipeline (30 reps, all scenarios)
    python -m sano_py --reps 5         # quick smoke run
    python -m sano_py --no-figures     # only run experiments + stats
"""
from __future__ import annotations
import argparse
import os
import sys
import time

from . import experiment, figures


DEFAULT_RESULTS = os.path.join("benchmarks", "results")
DEFAULT_FIGURES = os.path.join("paper", "figures")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="sano_py",
                                description="SANO simulation pipeline")
    p.add_argument("--reps", type=int, default=30,
                   help="repetitions per (system, scenario)")
    p.add_argument("--duration", type=int, default=600,
                   help="simulated seconds per run")
    p.add_argument("--results-dir", default=DEFAULT_RESULTS)
    p.add_argument("--figures-dir", default=DEFAULT_FIGURES)
    p.add_argument("--no-experiments", action="store_true")
    p.add_argument("--no-figures", action="store_true")
    args = p.parse_args(argv)

    t0 = time.time()
    if not args.no_experiments:
        print(f"[sano_py] running matrix: reps={args.reps}, "
              f"duration={args.duration}s")
        experiment.run_matrix(args.results_dir,
                              reps=args.reps,
                              duration_s=args.duration)
        print(f"[sano_py] wrote CSVs to {args.results_dir}")

    if not args.no_figures:
        print(f"[sano_py] generating figures into {args.figures_dir}")
        figures.make_all(args.results_dir, args.figures_dir)

    print(f"[sano_py] done in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())