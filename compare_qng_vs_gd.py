"""
compare_qng_vs_gd.py
====================
Head-to-head QNG vs Vanilla GD comparison with visualization.

Runs experiments (or loads pre-computed results) and generates:
  1. Convergence curves: cost vs iteration for both methods
  2. Success rate bar chart
  3. Iteration distribution histogram
  4. Wall-clock time comparison

Usage:
    # Run experiments + plot
    python experiments/compare_qng_vs_gd.py --n_trials 10 --plot

    # Load existing results + plot
    python experiments/compare_qng_vs_gd.py --load results/experiment_results_trapped_ion_*.json --plot

    # Quick test (5 trials, no plot)
    python experiments/compare_qng_vs_gd.py --n_trials 5
"""

import argparse
import json
import os
import sys
from glob import glob
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.run_experiments import run_experiment, EXPERIMENT_CONFIG


def load_results(path: str) -> Dict:
    """Load experiment results from a JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def compute_summary_stats(trials: List[Dict]) -> Dict:
    """Compute aggregate statistics over trials."""
    if not trials:
        return {}

    converged = [t["converged"] for t in trials]
    iters = [t["iterations"] for t in trials]
    conv_iters = [t["iterations"] for t in trials if t["converged"]]
    times = [t["wall_clock_time"] for t in trials]
    costs = [t["final_cost"] for t in trials]

    return {
        "success_rate": np.mean(converged) * 100,
        "mean_iterations": np.mean(iters),
        "std_iterations": np.std(iters),
        "mean_conv_iterations": np.mean(conv_iters) if conv_iters else float("nan"),
        "std_conv_iterations": np.std(conv_iters) if conv_iters else float("nan"),
        "mean_wall_clock_time": np.mean(times),
        "std_wall_clock_time": np.std(times),
        "mean_final_cost": np.mean(costs),
        "std_final_cost": np.std(costs),
        "n_trials": len(trials),
        "n_converged": sum(converged),
    }


def plot_results(results: Dict, save_dir: str = "results") -> None:
    """
    Generate comparison plots for QNG vs VGD.

    Creates:
        - convergence_curves.png   : Cost vs iteration (mean ± std)
        - success_rates.png        : Bar chart of convergence success rates
        - iteration_hist.png       : Distribution of iterations to converge
        - time_comparison.png      : Wall-clock time comparison

    Parameters
    ----------
    results : dict
        Output from run_experiment().
    save_dir : str
        Directory to save plots.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import seaborn as sns
    except ImportError:
        print("matplotlib/seaborn not available. Skipping plots.")
        return

    os.makedirs(save_dir, exist_ok=True)

    # Color scheme
    colors = {
        "QNG": "#2E86AB",       # Steel blue
        "VGD": "#E84855",       # Coral red
        "Adam": "#3BB273",      # Green
    }

    sns.set_style("whitegrid")
    plt.rcParams["font.size"] = 12
    plt.rcParams["figure.dpi"] = 120

    # -----------------------------------------------------------------------
    # Plot 1: Convergence curves (cost vs iteration)
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))

    for method, color in colors.items():
        trials = results.get(method, [])
        if not trials:
            continue

        # Align cost histories to same length
        max_len = max(len(t["cost_history"]) for t in trials)
        padded = []
        for t in trials:
            h = t["cost_history"]
            # Pad with last value
            padded.append(h + [h[-1]] * (max_len - len(h)))

        cost_matrix = np.array(padded)
        mean_cost = np.mean(cost_matrix, axis=0)
        std_cost = np.std(cost_matrix, axis=0)

        xs = np.arange(max_len)
        label = "QNG" if method == "QNG" else "Vanilla GD" if method == "VGD" else method
        ax.plot(xs, mean_cost, color=color, label=label, linewidth=2.0)
        ax.fill_between(
            xs,
            mean_cost - std_cost,
            mean_cost + std_cost,
            alpha=0.2,
            color=color,
        )

    platform = results.get("platform", "trapped_ion")
    ax.set_xlabel("Iteration", fontsize=13)
    ax.set_ylabel("Cost function ⟨H⟩", fontsize=13)
    ax.set_title(
        f"Convergence: QNG vs Vanilla GD\n"
        f"({platform} noise, {results.get('config', {}).get('n_trials', '?')} trials)",
        fontsize=14,
    )
    ax.legend(fontsize=12)
    ax.axhline(y=-3.3, color="black", linestyle="--", alpha=0.5, label="Optimal (-3.3)")
    fig.tight_layout()

    path = os.path.join(save_dir, "convergence_curves.png")
    fig.savefig(path)
    print(f"  ✓ Saved: {path}")
    plt.close(fig)

    # -----------------------------------------------------------------------
    # Plot 2: Success rate bar chart
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 4))

    methods_present = [m for m in ["VGD", "QNG"] if results.get(m)]
    success_rates = [np.mean([t["converged"] for t in results[m]]) * 100 for m in methods_present]
    labels = ["Vanilla GD" if m == "VGD" else "QNG" for m in methods_present]
    bar_colors = [colors[m] for m in methods_present]

    bars = ax.bar(labels, success_rates, color=bar_colors, edgecolor="white", linewidth=1.5)

    for bar, rate in zip(bars, success_rates):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{rate:.0f}%",
            ha="center", va="bottom", fontweight="bold", fontsize=13,
        )

    ax.set_ylim(0, 110)
    ax.set_ylabel("Convergence Success Rate (%)", fontsize=13)
    ax.set_title(f"Convergence Success Rate\n({platform} noise model)", fontsize=13)
    ax.axhline(y=100, color="gray", linestyle="--", alpha=0.4)
    fig.tight_layout()

    path = os.path.join(save_dir, "success_rates.png")
    fig.savefig(path)
    print(f"  ✓ Saved: {path}")
    plt.close(fig)

    # -----------------------------------------------------------------------
    # Plot 3: Iteration distribution (box plot)
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 4))

    data_to_plot = []
    tick_labels = []
    box_colors = []

    for method in ["VGD", "QNG"]:
        trials = results.get(method, [])
        if not trials:
            continue
        iters = [t["iterations"] for t in trials]
        data_to_plot.append(iters)
        tick_labels.append("Vanilla GD" if method == "VGD" else "QNG")
        box_colors.append(colors[method])

    bp = ax.boxplot(data_to_plot, patch_artist=True, notch=False)
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xticklabels(tick_labels, fontsize=12)
    ax.set_ylabel("Iterations", fontsize=13)
    ax.set_title(f"Iterations to Termination\n({platform} noise model)", fontsize=13)
    fig.tight_layout()

    path = os.path.join(save_dir, "iteration_distribution.png")
    fig.savefig(path)
    print(f"  ✓ Saved: {path}")
    plt.close(fig)

    # -----------------------------------------------------------------------
    # Plot 4: Wall-clock time
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 4))

    for i, method in enumerate(["VGD", "QNG"]):
        trials = results.get(method, [])
        if not trials:
            continue
        times = [t["wall_clock_time"] for t in trials]
        mean_t = np.mean(times)
        std_t = np.std(times)
        label = "Vanilla GD" if method == "VGD" else "QNG"
        ax.bar(i, mean_t, color=colors[method], yerr=std_t, capsize=8,
               edgecolor="white", linewidth=1.5)
        ax.text(i, mean_t + std_t + 1, f"{mean_t:.0f}s",
                ha="center", fontweight="bold", fontsize=12)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Vanilla GD", "QNG"], fontsize=12)
    ax.set_ylabel("Wall-clock Time (s)", fontsize=13)
    ax.set_title(f"Wall-clock Time to Solution\n({platform} noise model)", fontsize=13)
    fig.tight_layout()

    path = os.path.join(save_dir, "time_comparison.png")
    fig.savefig(path)
    print(f"  ✓ Saved: {path}")
    plt.close(fig)

    print(f"\nAll plots saved to: {save_dir}/")


def print_detailed_summary(results: Dict) -> None:
    """Print a detailed summary table."""
    print(f"\n{'='*72}")
    print(f"{'Detailed Experiment Summary':^72}")
    print(f"Platform: {results.get('platform', 'unknown')}")
    print(f"{'='*72}")

    for method in ["VGD", "QNG"]:
        trials = results.get(method, [])
        if not trials:
            continue
        stats = compute_summary_stats(trials)
        label = "Vanilla GD" if method == "VGD" else "QNG (Quantum Natural Gradient)"

        print(f"\n{label}:")
        print(f"  Trials              : {stats['n_trials']}")
        print(f"  Converged           : {stats['n_converged']}/{stats['n_trials']} "
              f"({stats['success_rate']:.1f}%)")
        print(f"  Mean iterations     : {stats['mean_iterations']:.1f} "
              f"± {stats['std_iterations']:.1f}")
        if not np.isnan(stats["mean_conv_iterations"]):
            print(f"  (converged only)    : {stats['mean_conv_iterations']:.1f} "
                  f"± {stats['std_conv_iterations']:.1f}")
        print(f"  Wall-clock time     : {stats['mean_wall_clock_time']:.1f} "
              f"± {stats['std_wall_clock_time']:.1f} s")
        print(f"  Final cost (mean)   : {stats['mean_final_cost']:.3f} "
              f"± {stats['std_final_cost']:.3f}")

    # Paper comparison
    print(f"\n{'Paper Results (Table 2)':^72}")
    print(f"  QNG convergence   : 95% | VGD convergence: 30% → 3.2× improvement")
    print(f"  QNG iterations    : 15  | VGD iterations : 95  → 6.3× fewer")
    print(f"  QNG time          : 287s| VGD time       : 342s → 16% faster")
    print(f"{'='*72}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare QNG vs Vanilla GD with plots",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--n_trials", type=int, default=10,
                        help="Number of trials (if running experiments)")
    parser.add_argument("--platform", type=str, default="trapped_ion",
                        help="Noise platform")
    parser.add_argument("--load", type=str, default=None,
                        help="Load results from JSON file (skip running experiments)")
    parser.add_argument("--plot", action="store_true",
                        help="Generate and save plots")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Output directory for plots and results")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    # Load or run experiments
    if args.load:
        paths = glob(args.load)
        if not paths:
            print(f"No files found matching: {args.load}")
            sys.exit(1)
        results = load_results(sorted(paths)[-1])   # Use most recent
        print(f"Loaded results from: {sorted(paths)[-1]}")
    else:
        results = run_experiment(
            n_trials=args.n_trials,
            platform=args.platform,
            output_dir=args.output_dir,
            verbose=args.verbose,
        )

    print_detailed_summary(results)

    if args.plot:
        print("\nGenerating plots...")
        plot_results(results, save_dir=args.output_dir)


if __name__ == "__main__":
    main()
