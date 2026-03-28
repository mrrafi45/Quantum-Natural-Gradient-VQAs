"""
run_experiments.py
==================
Full experimental suite for QNG vs Vanilla GD comparison.

Reproduces the 50-trial benchmark from Section II-III of the paper.

Usage:
    python experiments/run_experiments.py
    python experiments/run_experiments.py --n_trials 50 --platform trapped_ion
    python experiments/run_experiments.py --n_trials 5 --platform all --verbose

Outputs:
    results/experiment_results_{platform}_{timestamp}.json
    results/summary_table.csv (aggregate statistics)

Metrics collected per trial (Section II.C):
    - converged      : bool
    - iterations     : int
    - final_cost     : float
    - wall_clock_time: float (seconds)
    - cost_history   : List[float]
    - gradient_norms : List[float]
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np

# Add parent directory to path so we can import src modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ansatz import build_ansatz
from src.cost_function import MaxCutCostFunction
from src.noise_model import build_noise_model, PLATFORM_SPECS
from src.qng_optimizer import QNGOptimizer, OptimizationResult
from src.vanilla_gd import VanillaGradientDescent

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------

EXPERIMENT_CONFIG = {
    # Problem setup (Section II.B)
    "n_qubits": 4,
    "n_layers": 3,
    "edge_weights": {
        "(0, 1)": 1.5,
        "(0, 2)": 1.0,
        "(1, 3)": 0.8,
        "(2, 3)": 1.2,
    },

    # Optimization settings (Section II.C)
    "learning_rate": 0.1,
    "max_iterations": 100,
    "convergence_threshold": 0.1,
    "n_shots": 8192,

    # Initialization (Section II.C)
    "theta_init_range": 0.1,   # Uniform[-0.1, 0.1]

    # Statistical robustness
    "n_trials": 50,
}


def run_single_trial(
    seed: int,
    optimizer_name: str,
    platform: str,
    config: dict,
    verbose: bool = False,
) -> Dict:
    """
    Run a single optimization trial and return results as a dict.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.
    optimizer_name : str
        'QNG' or 'VGD'.
    platform : str
        Hardware platform for noise model.
    config : dict
        Experiment configuration.
    verbose : bool
        Print progress.

    Returns
    -------
    dict
        Trial result (JSON-serializable).
    """
    np.random.seed(seed)

    n_qubits = config["n_qubits"]
    n_layers = config["n_layers"]
    n_params = n_qubits * 2 * n_layers

    # Build problem components
    circuit = build_ansatz(n_qubits=n_qubits, n_layers=n_layers)

    edge_weights = {
        eval(k): v for k, v in config["edge_weights"].items()
    }
    cost_fn = MaxCutCostFunction(n_qubits=n_qubits, edge_weights=edge_weights)
    noise_model = build_noise_model(platform=platform)

    # Random initialization (paper: Uniform[-0.1, 0.1])
    theta_init = np.random.uniform(
        -config["theta_init_range"],
        config["theta_init_range"],
        n_params
    )

    # Build optimizer
    opt_kwargs = dict(
        learning_rate=config["learning_rate"],
        max_iterations=config["max_iterations"],
        convergence_threshold=config["convergence_threshold"],
        n_shots=config["n_shots"],
        verbose=verbose,
    )

    if optimizer_name == "QNG":
        optimizer = QNGOptimizer(
            **opt_kwargs,
            regularization=1e-4,
            qfim_mode="diagonal",  # Faster; change to "full" for exact paper replication
        )
    elif optimizer_name == "VGD":
        optimizer = VanillaGradientDescent(**opt_kwargs)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    # Run optimization
    result: OptimizationResult = optimizer.optimize(
        theta_init=theta_init,
        cost_fn=cost_fn,
        circuit=circuit,
        noise_model=noise_model,
        seed=seed,
    )

    return result.to_dict()


def run_experiment(
    n_trials: int = 50,
    platform: str = "trapped_ion",
    config: Optional[dict] = None,
    output_dir: str = "results",
    verbose: bool = False,
    base_seed: int = 42,
) -> Dict:
    """
    Run the full QNG vs VGD experiment suite.

    Parameters
    ----------
    n_trials : int
        Number of independent trials per optimizer.
    platform : str
        Hardware noise platform.
    config : dict, optional
        Override default experiment configuration.
    output_dir : str
        Directory for saving results.
    verbose : bool
        Print iteration details.
    base_seed : int
        Base random seed. Trial k uses seed (base_seed + k).

    Returns
    -------
    dict
        Full experiment results.
    """
    cfg = {**EXPERIMENT_CONFIG, **(config or {})}
    cfg["n_trials"] = n_trials

    print(f"\n{'='*70}")
    print(f"QNG vs Vanilla GD Experiment")
    print(f"{'='*70}")
    print(f"  Platform     : {platform} ({PLATFORM_SPECS[platform]['description']})")
    print(f"  Trials       : {n_trials}")
    print(f"  Qubits       : {cfg['n_qubits']}")
    print(f"  Layers       : {cfg['n_layers']}")
    print(f"  Parameters   : {cfg['n_qubits'] * 2 * cfg['n_layers']}")
    print(f"  Max iters    : {cfg['max_iterations']}")
    print(f"  Shots/eval   : {cfg['n_shots']}")
    print(f"{'='*70}\n")

    results = {
        "config": cfg,
        "platform": platform,
        "timestamp": datetime.now().isoformat(),
        "QNG": [],
        "VGD": [],
    }

    for optimizer_name in ["QNG", "VGD"]:
        print(f"\nRunning {optimizer_name} ({n_trials} trials)...")

        seeds = [base_seed + i for i in range(n_trials)]

        if HAS_TQDM:
            iter_obj = tqdm(seeds, desc=f"  {optimizer_name}", unit="trial")
        else:
            iter_obj = seeds

        for seed in iter_obj:
            trial_result = run_single_trial(
                seed=seed,
                optimizer_name=optimizer_name,
                platform=platform,
                config=cfg,
                verbose=verbose,
            )
            results[optimizer_name].append(trial_result)

        # Print per-optimizer summary
        trials = results[optimizer_name]
        success_rate = np.mean([t["converged"] for t in trials]) * 100
        conv_iters = [t["iterations"] for t in trials if t["converged"]]
        mean_iters = np.mean(conv_iters) if conv_iters else float("nan")
        mean_time = np.mean([t["wall_clock_time"] for t in trials])
        mean_cost = np.mean([t["final_cost"] for t in trials])

        print(f"  {optimizer_name} Summary:")
        print(f"    Success rate : {success_rate:.1f}%")
        print(f"    Mean iters   : {mean_iters:.1f}")
        print(f"    Mean time(s) : {mean_time:.1f}")
        print(f"    Mean cost    : {mean_cost:.3f}")

    # Print comparison table
    _print_comparison_table(results)

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"experiment_results_{platform}_{timestamp}.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to: {output_path}")

    return results


def _print_comparison_table(results: dict) -> None:
    """Print a formatted comparison table."""
    print(f"\n{'='*72}")
    print(f"{'QNG vs Vanilla GD: Summary':^72}")
    print(f"{'='*72}")
    print(f"{'Method':<16} {'Success Rate':>14} {'Mean Iters':>12} {'Mean Time(s)':>14} {'Final Cost':>12}")
    print(f"{'-'*72}")

    for method in ["VGD", "QNG"]:
        trials = results[method]
        if not trials:
            continue
        success_rate = np.mean([t["converged"] for t in trials]) * 100
        conv_iters = [t["iterations"] for t in trials if t["converged"]]
        mean_iters = np.mean(conv_iters) if conv_iters else float("nan")
        mean_time = np.mean([t["wall_clock_time"] for t in trials])
        mean_cost = np.mean([t["final_cost"] for t in trials])

        print(
            f"{'Vanilla GD' if method == 'VGD' else 'QNG':<16}"
            f" {success_rate:>13.1f}%"
            f" {mean_iters:>12.1f}"
            f" {mean_time:>14.1f}"
            f" {mean_cost:>12.3f}"
        )

    # Compute improvements
    qng_trials = results.get("QNG", [])
    vgd_trials = results.get("VGD", [])
    if qng_trials and vgd_trials:
        qng_sr = np.mean([t["converged"] for t in qng_trials])
        vgd_sr = np.mean([t["converged"] for t in vgd_trials])
        sr_ratio = qng_sr / max(vgd_sr, 1e-9)

        qng_iters = [t["iterations"] for t in qng_trials if t["converged"]]
        vgd_iters = [t["iterations"] for t in vgd_trials if t["converged"]]
        iter_ratio = (np.mean(vgd_iters) / np.mean(qng_iters)) if (qng_iters and vgd_iters) else float("nan")

        qng_time = np.mean([t["wall_clock_time"] for t in qng_trials])
        vgd_time = np.mean([t["wall_clock_time"] for t in vgd_trials])
        time_speedup = (vgd_time - qng_time) / vgd_time * 100

        print(f"{'-'*72}")
        print(f"{'Improvement':<16} {sr_ratio:>13.2f}× {iter_ratio:>12.2f}× {time_speedup:>13.1f}%")

    print(f"{'='*72}")


def main():
    parser = argparse.ArgumentParser(
        description="Run QNG vs Vanilla GD experiments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--n_trials", type=int, default=50,
        help="Number of independent trials per optimizer"
    )
    parser.add_argument(
        "--platform", type=str, default="trapped_ion",
        choices=list(PLATFORM_SPECS.keys()) + ["all"],
        help="Hardware noise platform"
    )
    parser.add_argument(
        "--output_dir", type=str, default="results",
        help="Directory to save results"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Base random seed"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print iteration-by-iteration details"
    )
    parser.add_argument(
        "--n_shots", type=int, default=8192,
        help="Number of measurement shots per circuit evaluation"
    )

    args = parser.parse_args()

    platforms = list(PLATFORM_SPECS.keys()) if args.platform == "all" else [args.platform]

    all_results = {}
    for platform in platforms:
        config_override = {"n_shots": args.n_shots}
        result = run_experiment(
            n_trials=args.n_trials,
            platform=platform,
            config=config_override,
            output_dir=args.output_dir,
            verbose=args.verbose,
            base_seed=args.seed,
        )
        all_results[platform] = result

    print("\n✓ All experiments complete.")
    return all_results


if __name__ == "__main__":
    main()
