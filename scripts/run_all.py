#!/usr/bin/env python3
"""
Run all experiments from the paper.

This reproduces all figures and tables:
  - Figure 1: Scaling law verification (Q1)
  - Figure 2: Separation theorem (Q2)
  - Figure 3 + Table 2: Benchmark comparison (Q3)
  - Figure 4: Fairness hierarchy (Q4)
  - Figures 5, 6: Ablation studies

Usage:
    python scripts/run_all.py [--seeds 0 1 2 3 4 5 6 7 8 9]
"""

import argparse
import os
import sys
import subprocess
import time


def run_script(script_name, args_str=""):
    """Run a script and report timing."""
    print(f"\n{'#'*70}")
    print(f"# Running: {script_name}")
    print(f"{'#'*70}\n")

    t0 = time.perf_counter()
    cmd = f"{sys.executable} scripts/{script_name} {args_str}"
    result = subprocess.run(cmd, shell=True, cwd=os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))

    elapsed = time.perf_counter() - t0
    status = "SUCCESS" if result.returncode == 0 else "FAILED"
    print(f"\n[{status}] {script_name} ({elapsed:.1f}s)")
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description='Run all experiments')
    parser.add_argument('--seeds', nargs='+', type=int,
                        default=list(range(10)))
    parser.add_argument('--skip_benchmarks', action='store_true',
                        help='Skip real-data benchmarks (requires downloads)')
    args = parser.parse_args()

    seeds_str = ' '.join(map(str, args.seeds))
    n_seeds = len(args.seeds)

    os.makedirs('results', exist_ok=True)

    print("=" * 70)
    print("The Price of Fairness in Active Learning")
    print("Reproducing all paper experiments")
    print(f"Seeds: {args.seeds}")
    print("=" * 70)

    t_total = time.perf_counter()
    results = {}

    # Q1: Scaling law
    results['scaling'] = run_script(
        'run_scaling.py', f'--n_seeds {n_seeds}'
    )

    # Q2: Separation
    results['separation'] = run_script(
        'run_separation.py', f'--n_seeds {n_seeds}'
    )

    # Q3: Benchmarks (requires data downloads)
    if not args.skip_benchmarks:
        results['benchmarks'] = run_script(
            'run_benchmarks.py', f'--seeds {seeds_str}'
        )
    else:
        print("\nSkipping benchmark experiments (--skip_benchmarks)")

    # Q4: Hierarchy
    results['hierarchy'] = run_script(
        'run_hierarchy.py', f'--n_seeds {n_seeds}'
    )

    # Ablation studies
    results['ablation'] = run_script(
        'run_ablation.py', f'--n_seeds {n_seeds}'
    )

    # Summary
    total_time = time.perf_counter() - t_total
    print(f"\n{'='*70}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*70}")
    for name, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {status}  {name}")
    print(f"\nTotal time: {total_time:.1f}s ({total_time/60:.1f}min)")
    print(f"Results saved to: results/")


if __name__ == '__main__':
    main()
