"""
plots.py
========
Visualization utilities for QNG vs Vanilla GD experiment results.

Generates all figures from the paper:
  - Fig 1: Barren plateau — cost landscape flattening with qubit count
  - Fig 2: Gradient std decay with system size
  - Fig 3: Parameter space geometry: VGD vs QNG
  - Fig 4: Convergence curves: QNG vs VGD (mean ± std over 50 trials)
  - Fig 5: Success rate comparison across noise models
  - Fig 6: QFIM condition number evolution

Usage:
    python results/plots.py --input results/sample_output.json
    python results/plots.py --input results/experiment_results_trapped_ion_*.json --all
"""

import argparse
import json
import os
import sys
from glob import glob
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Matplotlib setup
# ---------------------------------------------------------------------------

def _setup_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")   # Non-interactive backend for scripts
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        import seaborn as sns
        sns.set_style("whitegrid")
        plt.rcParams.update({
            "font.size": 12,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "legend.fontsize": 11,
            "figure.dpi": 120,
            "savefig.bbox": "tight",
            "savefig.dpi": 150,
        })
        return plt, gridspec, sns
    except ImportError as e:
        raise ImportError(
            f"Visualization requires matplotlib and seaborn.\n"
            f"Install with: pip install matplotlib seaborn\n"
            f"Error: {e}"
        )


# ---------------------------------------------------------------------------
# Color palette (consistent across all figures)
# ---------------------------------------------------------------------------

PALETTE = {
    "QNG":  "#2E86AB",   # Steel blue
    "VGD":  "#E84855",   # Coral red
    "Adam": "#3BB273",   # Forest green
    "optimal": "#555555",
    "fill_alpha": 0.18,
}

METHOD_LABELS = {
    "QNG": "QNG (Quantum Natural Gradient)",
    "VGD": "Vanilla Gradient Descent",
    "Adam": "Adam",
}


# ---------------------------------------------------------------------------
# Helper: align cost histories to same length
# ---------------------------------------------------------------------------

def _pad_histories(trials: List[Dict], key: str = "cost_history") -> np.ndarray:
    """Pad histories with last value to make uniform length array."""
    histories = [t[key] for t in trials if t.get(key)]
    if not histories:
        return np.array([])
    max_len = max(len(h) for h in histories)
    padded = np.array([h + [h[-1]] * (max_len - len(h)) for h in histories])
    return padded


def _compute_stats(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (mean, mean-std, mean+std) across rows."""
    mean = np.mean(matrix, axis=0)
    std = np.std(matrix, axis=0)
    return mean, mean - std, mean + std


# ---------------------------------------------------------------------------
# Figure 1: Convergence curves (main result figure)
# ---------------------------------------------------------------------------

def plot_convergence_curves(
    results: Dict,
    save_path: str = "results/convergence_curves.png",
    show_optimal: bool = True,
) -> None:
    """
    Plot mean cost vs iteration for QNG and VGD with ±1 std band.

    This is the central result figure of the paper.
    """
    plt, _, _ = _setup_matplotlib()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5.5))

    for method in ["VGD", "QNG"]:
        trials = results.get(method, [])
        if not trials:
            continue

        cost_matrix = _pad_histories(trials, "cost_history")
        if cost_matrix.size == 0:
            continue

        mean, lo, hi = _compute_stats(cost_matrix)
        xs = np.arange(len(mean))
        color = PALETTE[method]
        label = METHOD_LABELS[method]

        ax.plot(xs, mean, color=color, label=label, linewidth=2.2, zorder=3)
        ax.fill_between(xs, lo, hi, alpha=PALETTE["fill_alpha"], color=color, zorder=2)

    if show_optimal:
        ax.axhline(
            y=-3.3, color=PALETTE["optimal"],
            linestyle="--", linewidth=1.4, alpha=0.7,
            label="Optimal: −3.3 (classical)"
        )

    n_trials = results.get("config", {}).get("n_trials", "?")
    platform = results.get("platform", "trapped_ion")
    ax.set_xlabel("Iteration", fontsize=13)
    ax.set_ylabel("Cost function ⟨ψ(θ)|H|ψ(θ)⟩", fontsize=13)
    ax.set_title(
        f"Convergence Curves: QNG vs Vanilla GD\n"
        f"4-qubit MaxCut · {platform} noise · {n_trials} trials · shaded = ±1 std",
        fontsize=13,
    )
    ax.legend(loc="lower left", fontsize=11)
    ax.set_xlim(left=0)

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  ✓ {save_path}")


# ---------------------------------------------------------------------------
# Figure 2: Success rate bar chart
# ---------------------------------------------------------------------------

def plot_success_rates(
    results_by_platform: Dict[str, Dict],
    save_path: str = "results/success_rates.png",
) -> None:
    """
    Bar chart of convergence success rates for QNG vs VGD across platforms.

    Parameters
    ----------
    results_by_platform : dict
        Keys are platform names, values are results dicts.
    """
    plt, _, _ = _setup_matplotlib()
    import matplotlib.pyplot as plt

    platforms = list(results_by_platform.keys())
    n_platforms = len(platforms)

    # Paper labels
    platform_labels = {
        "trapped_ion": "Trapped-Ion\n(IonQ Forte)",
        "superconducting": "Superconducting\n(IBM Falcon)",
        "high_noise": "High-Noise\n(Degraded HW)",
        "noiseless": "Noiseless\n(Ideal)",
    }

    x = np.arange(n_platforms)
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(7, 2.5 * n_platforms), 5))

    for i, method in enumerate(["VGD", "QNG"]):
        rates = []
        for platform in platforms:
            trials = results_by_platform[platform].get(method, [])
            rate = np.mean([t["converged"] for t in trials]) * 100 if trials else 0
            rates.append(rate)

        offset = (i - 0.5) * width
        bars = ax.bar(
            x + offset, rates,
            width=width,
            color=PALETTE[method],
            label=METHOD_LABELS[method],
            edgecolor="white", linewidth=1.2,
        )
        for bar, rate in zip(bars, rates):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.5,
                f"{rate:.0f}%",
                ha="center", va="bottom", fontsize=10, fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels([platform_labels.get(p, p) for p in platforms], fontsize=11)
    ax.set_ylim(0, 115)
    ax.set_ylabel("Convergence Success Rate (%)", fontsize=12)
    ax.set_title("QNG vs Vanilla GD: Convergence Success Rate by Noise Model", fontsize=13)
    ax.legend(fontsize=11)
    ax.axhline(y=100, color="gray", linestyle="--", alpha=0.4, linewidth=1.0)

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  ✓ {save_path}")


# ---------------------------------------------------------------------------
# Figure 3: Gradient norm evolution
# ---------------------------------------------------------------------------

def plot_gradient_norms(
    results: Dict,
    save_path: str = "results/gradient_norms.png",
) -> None:
    """Plot evolution of ||∇L|| over iterations — shows barren plateau effect."""
    plt, _, _ = _setup_matplotlib()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))

    for method in ["VGD", "QNG"]:
        trials = results.get(method, [])
        if not trials:
            continue

        grad_matrix = _pad_histories(trials, "gradient_norm_history")
        if grad_matrix.size == 0:
            continue

        mean, lo, hi = _compute_stats(grad_matrix)
        xs = np.arange(len(mean))
        color = PALETTE[method]
        ax.semilogy(xs, mean, color=color, label=METHOD_LABELS[method], linewidth=2.0)
        ax.fill_between(xs, np.maximum(lo, 1e-9), hi,
                        alpha=PALETTE["fill_alpha"], color=color)

    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Gradient Norm ‖∇L‖ (log scale)", fontsize=12)
    ax.set_title(
        "Gradient Norm Evolution\n"
        "QNG maintains larger gradients — resists barren plateaus",
        fontsize=13,
    )
    ax.legend(fontsize=11)
    ax.set_xlim(left=0)

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  ✓ {save_path}")


# ---------------------------------------------------------------------------
# Figure 4: QFIM condition number evolution
# ---------------------------------------------------------------------------

def plot_qfim_condition(
    results: Dict,
    save_path: str = "results/qfim_condition.png",
) -> None:
    """
    Plot QFIM condition number κ(QFIM) over iterations.

    High κ → ill-conditioned QFIM → regularization kicks in.
    Shows how noise degrades QFIM geometry.
    """
    plt, _, _ = _setup_matplotlib()
    import matplotlib.pyplot as plt

    qng_trials = results.get("QNG", [])
    if not qng_trials:
        print("  No QNG results with QFIM condition data. Skipping.")
        return

    cond_matrix = _pad_histories(qng_trials, "qfim_condition_history")
    if cond_matrix.size == 0:
        print("  No QFIM condition history found. Skipping.")
        return

    fig, ax = plt.subplots(figsize=(8, 4.5))

    mean, lo, hi = _compute_stats(cond_matrix)
    xs = np.arange(len(mean))

    ax.semilogy(xs, mean, color=PALETTE["QNG"], linewidth=2.0,
                label="Mean κ(QFIM)")
    ax.fill_between(xs, np.maximum(lo, 1.0), hi,
                    alpha=PALETTE["fill_alpha"], color=PALETTE["QNG"])
    ax.axhline(y=1e6, color="red", linestyle="--", alpha=0.6,
               label="Regularization threshold (10⁶)")

    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("QFIM Condition Number κ(QFIM) (log scale)", fontsize=12)
    ax.set_title(
        "QFIM Condition Number Over Training\n"
        "Regularization λI added when κ exceeds threshold",
        fontsize=13,
    )
    ax.legend(fontsize=11)
    ax.set_xlim(left=0)

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  ✓ {save_path}")


# ---------------------------------------------------------------------------
# Figure 5: Barren plateau illustration (synthetic)
# ---------------------------------------------------------------------------

def plot_barren_plateau_illustration(
    save_path: str = "results/barren_plateau_illustration.png",
) -> None:
    """
    Synthetic illustration of the barren plateau phenomenon.

    Shows how gradient variance decays exponentially with qubit count,
    mirroring Fig 2 from the paper.
    """
    plt, _, _ = _setup_matplotlib()
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Panel A: Gradient variance vs qubit count
    ax = axes[0]
    qubit_counts = np.arange(2, 15)

    # Theoretical: Var[∂C/∂θ] ~ 2^{-n} (global cost)
    var_global = 2.0 ** (-qubit_counts)
    var_local = qubit_counts ** (-2)    # Polynomial decay (local cost)

    ax.semilogy(qubit_counts, var_global / var_global[0], "o-",
                color=PALETTE["VGD"], linewidth=2.0, markersize=6,
                label="Global cost (exponential decay)")
    ax.semilogy(qubit_counts, var_local / var_local[0], "s--",
                color=PALETTE["QNG"], linewidth=2.0, markersize=6,
                label="Local cost (polynomial decay)")

    ax.set_xlabel("Number of Qubits n", fontsize=12)
    ax.set_ylabel("Normalized Gradient Variance", fontsize=12)
    ax.set_title("Barren Plateau: Gradient Variance Decay\nvs System Size", fontsize=13)
    ax.legend(fontsize=10)
    ax.set_xticks(qubit_counts)

    # Panel B: Cost landscape (2D slice through parameter space)
    ax2 = axes[1]
    theta1 = np.linspace(-np.pi, np.pi, 200)
    theta2 = np.linspace(-np.pi, np.pi, 200)
    T1, T2 = np.meshgrid(theta1, theta2)

    # Large system: nearly flat landscape
    Z_flat = 0.01 * np.sin(T1) * np.cos(T2) + 0.005 * np.random.default_rng(0).standard_normal((200,200))
    # Small system: meaningful landscape
    Z_curved = np.sin(T1) * np.cos(T2) * np.exp(-0.1 * (T1**2 + T2**2))

    # Show both as side-by-side contour insets using subgridspec
    im = ax2.contourf(T1, T2, Z_flat, levels=30, cmap="RdBu_r")
    ax2.set_xlabel("Parameter θ₁", fontsize=12)
    ax2.set_ylabel("Parameter θ₂", fontsize=12)
    ax2.set_title("Cost Landscape (Large System)\n→ Nearly Flat (Barren Plateau)", fontsize=13)
    fig.colorbar(im, ax=ax2, label="⟨H⟩")

    fig.suptitle("The Barren Plateau Phenomenon (McClean et al. 2018)", fontsize=14, y=1.02)
    fig.tight_layout()

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  ✓ {save_path}")


# ---------------------------------------------------------------------------
# Figure 6: Per-iteration cost comparison
# ---------------------------------------------------------------------------

def plot_per_iteration_cost_comparison(
    results: Dict,
    save_path: str = "results/per_iteration_cost.png",
) -> None:
    """
    Box plots of wall-clock time and final cost.
    Shows that QNG is cheaper per solution despite higher per-iteration cost.
    """
    plt, _, _ = _setup_matplotlib()
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # Panel A: Wall-clock time
    ax = axes[0]
    times_data = []
    tick_labels = []
    colors_list = []

    for method in ["VGD", "QNG"]:
        trials = results.get(method, [])
        if trials:
            times_data.append([t["wall_clock_time"] for t in trials])
            tick_labels.append("Vanilla GD" if method == "VGD" else "QNG")
            colors_list.append(PALETTE[method])

    bp = ax.boxplot(times_data, patch_artist=True, notch=False, widths=0.5)
    for patch, color in zip(bp["boxes"], colors_list):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    for element in ["whiskers", "caps", "medians", "fliers"]:
        for item in bp[element]:
            item.set(color="black", linewidth=1.2)

    ax.set_xticklabels(tick_labels, fontsize=11)
    ax.set_ylabel("Wall-clock Time (s)", fontsize=12)
    ax.set_title("Wall-clock Time to Termination", fontsize=13)

    # Panel B: Final cost
    ax2 = axes[1]
    cost_data = []
    tick_labels2 = []
    colors_list2 = []

    for method in ["VGD", "QNG"]:
        trials = results.get(method, [])
        if trials:
            cost_data.append([t["final_cost"] for t in trials])
            tick_labels2.append("Vanilla GD" if method == "VGD" else "QNG")
            colors_list2.append(PALETTE[method])

    bp2 = ax2.boxplot(cost_data, patch_artist=True, notch=False, widths=0.5)
    for patch, color in zip(bp2["boxes"], colors_list2):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    for element in ["whiskers", "caps", "medians", "fliers"]:
        for item in bp2[element]:
            item.set(color="black", linewidth=1.2)

    ax2.axhline(y=-3.3, color=PALETTE["optimal"], linestyle="--",
                alpha=0.7, label="Optimal (−3.3)")
    ax2.set_xticklabels(tick_labels2, fontsize=11)
    ax2.set_ylabel("Final Cost ⟨H⟩", fontsize=12)
    ax2.set_title("Final Solution Quality", fontsize=13)
    ax2.legend(fontsize=10)

    fig.suptitle(
        f"QNG vs Vanilla GD — Time & Solution Quality\n"
        f"({results.get('platform','trapped_ion')} noise · "
        f"{results.get('config',{}).get('n_trials','?')} trials)",
        fontsize=13,
    )
    fig.tight_layout()

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  ✓ {save_path}")


# ---------------------------------------------------------------------------
# Generate all plots
# ---------------------------------------------------------------------------

def generate_all_plots(results: Dict, output_dir: str = "results") -> None:
    """Generate every figure for the given results dict."""
    print(f"\nGenerating plots → {output_dir}/")

    plot_convergence_curves(results, save_path=f"{output_dir}/convergence_curves.png")
    plot_gradient_norms(results, save_path=f"{output_dir}/gradient_norms.png")
    plot_qfim_condition(results, save_path=f"{output_dir}/qfim_condition.png")
    plot_per_iteration_cost_comparison(results, save_path=f"{output_dir}/per_iteration_cost.png")
    plot_success_rates(
        {results.get("platform", "trapped_ion"): results},
        save_path=f"{output_dir}/success_rates.png",
    )
    plot_barren_plateau_illustration(save_path=f"{output_dir}/barren_plateau_illustration.png")

    print(f"\n✓ All plots saved to: {output_dir}/")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate QNG experiment plots",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", type=str, default="results/sample_output.json",
        help="Path to experiment JSON results (glob patterns accepted)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="results",
        help="Directory to save plots",
    )
    parser.add_argument(
        "--barren_plateau_only", action="store_true",
        help="Only generate the barren plateau illustration",
    )
    args = parser.parse_args()

    if args.barren_plateau_only:
        plot_barren_plateau_illustration(
            save_path=os.path.join(args.output_dir, "barren_plateau_illustration.png")
        )
        return

    paths = glob(args.input)
    if not paths:
        print(f"No files found: {args.input}")
        sys.exit(1)

    results_path = sorted(paths)[-1]
    print(f"Loading: {results_path}")
    with open(results_path) as f:
        results = json.load(f)

    generate_all_plots(results, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
