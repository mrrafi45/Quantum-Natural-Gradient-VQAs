"""
qng_optimizer.py
================
Quantum Natural Gradient (QNG) optimizer implementation.

The QNG update rule preconditions gradient descent with the inverse
Quantum Fisher Information Matrix (QFIM):

    θ(k+1) = θ(k) - η × QFIM(k)⁻¹ × ∇L(θ(k))

This follows geodesics on the Riemannian manifold of quantum states,
automatically rescaling step sizes per direction based on local curvature.

Key differences from vanilla gradient descent:
  - Step size is geometry-aware (large steps in flat regions, small in curved)
  - Automatically down-weights barren plateau directions
  - More expensive per iteration: O(P²) vs O(P) circuit evaluations
  - More robust to noise and ill-conditioning via regularization

Algorithm (Section II.C of the paper):
    1. Initialize θ ∈ [-0.1, 0.1]^P uniformly at random
    2. At each iteration k:
       a. Compute gradient ∇L(θ) via parameter-shift rule  [2P evaluations]
       b. Compute QFIM(θ) via parameter-shift rule          [2P(P+1) evaluations]
       c. Regularize and invert QFIM                        [O(P³) classical]
       d. Update: θ ← θ - η × QFIM⁻¹ × ∇L(θ)
       e. Evaluate cost C(θ)
    3. Stop if C(θ) < convergence_threshold OR k ≥ max_iterations

Reference:
    Stokes et al. (2020) — Quantum Natural Gradient. Quantum 4, 269.
    Cerezo et al. (2023) — Cost function structure and trainability.
"""

import time
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer.noise import NoiseModel

from src.cost_function import BaseCostFunction
from src.qfim import compute_qfim, compute_qfim_diagonal, invert_qfim


# ---------------------------------------------------------------------------
# Result dataclass (returned by optimizer)
# ---------------------------------------------------------------------------

@dataclass
class OptimizationResult:
    """Holds the complete result of an optimization run."""
    method: str                             # 'QNG' or 'VGD'
    converged: bool                         # Did it reach the threshold?
    iterations: int                         # Number of iterations performed
    final_cost: float                       # Final cost function value
    wall_clock_time: float                  # Total time in seconds
    cost_history: List[float] = field(default_factory=list)
    gradient_norm_history: List[float] = field(default_factory=list)
    qfim_condition_history: List[float] = field(default_factory=list)
    theta_final: Optional[np.ndarray] = None
    seed: Optional[int] = None

    def to_dict(self) -> Dict:
        """Serialize to a plain dictionary (JSON-compatible)."""
        return {
            "method": self.method,
            "converged": self.converged,
            "iterations": self.iterations,
            "final_cost": self.final_cost,
            "wall_clock_time": self.wall_clock_time,
            "cost_history": self.cost_history,
            "gradient_norm_history": self.gradient_norm_history,
            "qfim_condition_history": self.qfim_condition_history,
            "theta_final": self.theta_final.tolist() if self.theta_final is not None else None,
            "seed": self.seed,
        }


# ---------------------------------------------------------------------------
# QNG Optimizer
# ---------------------------------------------------------------------------

class QNGOptimizer:
    """
    Quantum Natural Gradient Optimizer.

    Uses the QFIM as a Riemannian metric to precondition gradient updates.
    Includes adaptive regularization to handle ill-conditioned QFIM under
    NISQ noise.

    Parameters
    ----------
    learning_rate : float
        Step size η. Paper uses 0.1. Default: 0.1.
    regularization : float
        Base Tikhonov regularization λ for QFIM inversion. Default: 1e-4.
    max_iterations : int
        Maximum optimization steps. Default: 100.
    convergence_threshold : float
        Stop when cost < this value. Paper uses 0.1. Default: 0.1.
    n_shots : int
        Shots per circuit evaluation. Paper uses 8192. Default: 8192.
    qfim_mode : str
        'full' — Full QFIM with O(P²) evaluations (paper default)
        'diagonal' — Diagonal-only QFIM with O(P) evaluations (faster approx)
    verbose : bool
        If True, print iteration-by-iteration progress.

    Example
    -------
    >>> from src.qng_optimizer import QNGOptimizer
    >>> optimizer = QNGOptimizer(learning_rate=0.1, max_iterations=100)
    >>> result = optimizer.optimize(theta_init, cost_fn, circuit, noise_model)
    >>> print(f"Converged in {result.iterations} iterations")
    """

    def __init__(
        self,
        learning_rate: float = 0.1,
        regularization: float = 1e-4,
        max_iterations: int = 100,
        convergence_threshold: float = 0.1,
        n_shots: int = 8192,
        qfim_mode: str = "full",
        verbose: bool = False,
    ):
        self.lr = learning_rate
        self.regularization = regularization
        self.max_iter = max_iterations
        self.conv_threshold = convergence_threshold
        self.n_shots = n_shots
        self.qfim_mode = qfim_mode
        self.verbose = verbose

        if qfim_mode not in ("full", "diagonal"):
            raise ValueError("qfim_mode must be 'full' or 'diagonal'.")

    def optimize(
        self,
        theta_init: np.ndarray,
        cost_fn: BaseCostFunction,
        circuit: QuantumCircuit,
        noise_model: Optional[NoiseModel] = None,
        seed: Optional[int] = None,
    ) -> OptimizationResult:
        """
        Run QNG optimization starting from theta_init.

        Parameters
        ----------
        theta_init : np.ndarray
            Initial parameter vector, shape (n_params,).
        cost_fn : BaseCostFunction
            The cost function to minimize (e.g., MaxCutCostFunction).
        circuit : QuantumCircuit
            Parameterized ansatz circuit.
        noise_model : NoiseModel, optional
            NISQ noise model. If None, ideal simulation.
        seed : int, optional
            Random seed (stored in result for reproducibility).

        Returns
        -------
        OptimizationResult
        """
        theta = theta_init.copy()
        n_params = len(theta)

        cost_history = []
        grad_norm_history = []
        qfim_cond_history = []

        t_start = time.time()
        converged = False
        final_cost = None

        for iteration in range(self.max_iter):
            # ----------------------------------------------------------------
            # Step 1: Evaluate current cost
            # ----------------------------------------------------------------
            current_cost = cost_fn.evaluate(theta, circuit, noise_model, self.n_shots)
            cost_history.append(current_cost)

            if self.verbose:
                print(f"  [QNG] Iter {iteration:3d}  cost={current_cost:+.4f}")

            # Check convergence: cost below threshold means we found a good solution
            # Cost function is negative (we minimize energy), so check |cost| > threshold
            if abs(current_cost - cost_fn.optimal_value) < self.conv_threshold:
                converged = True
                final_cost = current_cost
                if self.verbose:
                    print(f"  [QNG] Converged at iteration {iteration}!")
                break

            # ----------------------------------------------------------------
            # Step 2: Compute gradient via parameter-shift rule
            # ----------------------------------------------------------------
            gradient = cost_fn.compute_gradient(theta, circuit, noise_model, self.n_shots)
            grad_norm = float(np.linalg.norm(gradient))
            grad_norm_history.append(grad_norm)

            # ----------------------------------------------------------------
            # Step 3: Compute QFIM
            # ----------------------------------------------------------------
            if self.qfim_mode == "full":
                qfim = compute_qfim(theta, circuit, noise_model, self.n_shots)
                cond_num = float(np.linalg.cond(qfim + self.regularization * np.eye(n_params)))
            else:
                # Diagonal approximation
                qfim_diag = compute_qfim_diagonal(theta, circuit, noise_model, self.n_shots)
                qfim = np.diag(qfim_diag)
                cond_num = float(np.max(qfim_diag) / (np.min(qfim_diag) + 1e-12))

            qfim_cond_history.append(cond_num)

            # ----------------------------------------------------------------
            # Step 4: Invert QFIM with regularization
            # ----------------------------------------------------------------
            try:
                qfim_inv = invert_qfim(qfim, self.regularization, method="lu")
            except Exception as e:
                warnings.warn(
                    f"QFIM inversion failed at iter {iteration}: {e}. "
                    "Using pseudoinverse fallback."
                )
                qfim_inv = invert_qfim(qfim, self.regularization * 100, method="pinv")

            # ----------------------------------------------------------------
            # Step 5: Compute QNG update direction
            # ----------------------------------------------------------------
            natural_gradient = qfim_inv @ gradient

            # Clip update to avoid exploding steps
            max_step = 1.0
            step_norm = np.linalg.norm(natural_gradient)
            if step_norm > max_step:
                natural_gradient = natural_gradient * (max_step / step_norm)

            # ----------------------------------------------------------------
            # Step 6: Parameter update
            # ----------------------------------------------------------------
            theta = theta - self.lr * natural_gradient

        else:
            # Loop completed without convergence
            final_cost = cost_fn.evaluate(theta, circuit, noise_model, self.n_shots)
            cost_history.append(final_cost)

        if final_cost is None:
            final_cost = current_cost

        wall_clock_time = time.time() - t_start

        return OptimizationResult(
            method="QNG",
            converged=converged,
            iterations=len(cost_history) - 1,
            final_cost=final_cost,
            wall_clock_time=wall_clock_time,
            cost_history=cost_history,
            gradient_norm_history=grad_norm_history,
            qfim_condition_history=qfim_cond_history,
            theta_final=theta,
            seed=seed,
        )


if __name__ == "__main__":
    import numpy as np
    from src.ansatz import build_ansatz
    from src.cost_function import MaxCutCostFunction
    from src.noise_model import build_noise_model

    print("QNG Optimizer — Single Trial Demo")
    print("=" * 50)

    np.random.seed(42)
    n_qubits, n_layers = 4, 3
    n_params = n_qubits * 2 * n_layers

    circuit = build_ansatz(n_qubits=n_qubits, n_layers=n_layers)
    cost_fn = MaxCutCostFunction(n_qubits=n_qubits)
    noise_model = build_noise_model("trapped_ion")

    theta_init = np.random.uniform(-0.1, 0.1, n_params)

    optimizer = QNGOptimizer(
        learning_rate=0.1,
        max_iterations=20,   # Short demo
        convergence_threshold=0.1,
        n_shots=1024,
        qfim_mode="diagonal",   # Faster for demo
        verbose=True,
    )

    result = optimizer.optimize(theta_init, cost_fn, circuit, noise_model, seed=42)

    print(f"\nResult:")
    print(f"  Converged     : {result.converged}")
    print(f"  Iterations    : {result.iterations}")
    print(f"  Final cost    : {result.final_cost:.4f}")
    print(f"  Wall-clock    : {result.wall_clock_time:.2f}s")
