"""
vanilla_gd.py
=============
Vanilla Gradient Descent (VGD) baseline optimizer for NISQ VQAs.

Implements the standard gradient descent update:

    θ(k+1) = θ(k) - η × ∇L(θ(k))

Gradient is computed using the parameter-shift rule (same as QNG),
so comparison is fair — both optimizers access the same gradient
information. The only difference is that VGD does NOT use QFIM.

This is the baseline that QNG outperforms in the paper:
  - VGD convergence success rate: 30% (vs 95% for QNG)
  - VGD mean iterations to convergence: 95 (vs 15 for QNG)
  - VGD wall-clock time: 342 s (vs 287 s for QNG)

Algorithm (Section II.C of the paper — VGD setup):
    1. Initialize θ ∈ [-0.1, 0.1]^P uniformly at random
    2. At each iteration k:
       a. Compute ∇L(θ) via parameter-shift rule  [2P circuit evaluations]
       b. Update: θ ← θ - η × ∇L(θ)
       c. Evaluate cost C(θ)
    3. Stop if C(θ) < convergence_threshold OR k ≥ max_iterations
"""

import time
import warnings
from typing import Optional

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer.noise import NoiseModel

from src.cost_function import BaseCostFunction
from src.qng_optimizer import OptimizationResult


class VanillaGradientDescent:
    """
    Vanilla Gradient Descent optimizer for variational quantum circuits.

    Uses the parameter-shift rule for gradient computation — identical
    to QNG — so differences in performance are attributable solely to
    the absence of QFIM preconditioning.

    Parameters
    ----------
    learning_rate : float
        Step size η. Paper uses 0.1 for both VGD and QNG. Default: 0.1.
    max_iterations : int
        Maximum steps. Default: 100.
    convergence_threshold : float
        Convergence criterion: stop when |cost - optimal| < threshold.
        Default: 0.1.
    n_shots : int
        Shots per circuit evaluation. Paper uses 8192. Default: 8192.
    momentum : float
        Momentum factor (0.0 = no momentum). Included for extension. Default: 0.0.
    verbose : bool
        Print iteration-by-iteration progress. Default: False.

    Example
    -------
    >>> from src.vanilla_gd import VanillaGradientDescent
    >>> optimizer = VanillaGradientDescent(learning_rate=0.1, max_iterations=100)
    >>> result = optimizer.optimize(theta_init, cost_fn, circuit, noise_model)
    >>> print(f"Converged: {result.converged}, Iterations: {result.iterations}")
    """

    def __init__(
        self,
        learning_rate: float = 0.1,
        max_iterations: int = 100,
        convergence_threshold: float = 0.1,
        n_shots: int = 8192,
        momentum: float = 0.0,
        verbose: bool = False,
    ):
        self.lr = learning_rate
        self.max_iter = max_iterations
        self.conv_threshold = convergence_threshold
        self.n_shots = n_shots
        self.momentum = momentum
        self.verbose = verbose

        if not 0.0 <= momentum < 1.0:
            raise ValueError("momentum must be in [0.0, 1.0).")

    def optimize(
        self,
        theta_init: np.ndarray,
        cost_fn: BaseCostFunction,
        circuit: QuantumCircuit,
        noise_model: Optional[NoiseModel] = None,
        seed: Optional[int] = None,
    ) -> OptimizationResult:
        """
        Run vanilla gradient descent starting from theta_init.

        Parameters
        ----------
        theta_init : np.ndarray
            Initial parameter vector, shape (n_params,).
        cost_fn : BaseCostFunction
            Cost function to minimize.
        circuit : QuantumCircuit
            Parameterized ansatz circuit.
        noise_model : NoiseModel, optional
            NISQ noise model. If None, ideal simulation.
        seed : int, optional
            Stored in result for traceability.

        Returns
        -------
        OptimizationResult
            Full optimization trace including cost and gradient history.
        """
        theta = theta_init.copy()

        cost_history = []
        grad_norm_history = []
        velocity = np.zeros_like(theta)  # For momentum (0 if momentum=0.0)

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
                print(f"  [VGD] Iter {iteration:3d}  cost={current_cost:+.4f}")

            # Check convergence
            if abs(current_cost - cost_fn.optimal_value) < self.conv_threshold:
                converged = True
                final_cost = current_cost
                if self.verbose:
                    print(f"  [VGD] Converged at iteration {iteration}!")
                break

            # ----------------------------------------------------------------
            # Step 2: Compute gradient via parameter-shift rule
            # ----------------------------------------------------------------
            gradient = cost_fn.compute_gradient(theta, circuit, noise_model, self.n_shots)
            grad_norm = float(np.linalg.norm(gradient))
            grad_norm_history.append(grad_norm)

            # ----------------------------------------------------------------
            # Step 3: Gradient descent update (with optional momentum)
            # ----------------------------------------------------------------
            velocity = self.momentum * velocity + gradient
            theta = theta - self.lr * velocity

        else:
            # Loop completed without convergence
            final_cost = cost_fn.evaluate(theta, circuit, noise_model, self.n_shots)
            cost_history.append(final_cost)

        if final_cost is None:
            final_cost = current_cost

        wall_clock_time = time.time() - t_start

        return OptimizationResult(
            method="VGD",
            converged=converged,
            iterations=len(cost_history) - 1,
            final_cost=final_cost,
            wall_clock_time=wall_clock_time,
            cost_history=cost_history,
            gradient_norm_history=grad_norm_history,
            qfim_condition_history=[],   # VGD doesn't compute QFIM
            theta_final=theta,
            seed=seed,
        )


# ---------------------------------------------------------------------------
# Adam optimizer for additional comparison (optional baseline)
# ---------------------------------------------------------------------------

class AdamOptimizer:
    """
    Adam optimizer as an additional classical baseline.

    Adam adapts per-parameter learning rates using first and second
    moment estimates of the gradient. It is a stronger baseline than
    vanilla GD but still doesn't use quantum geometric information (QFIM).

    Parameters
    ----------
    learning_rate : float
        Base learning rate. Default: 0.01.
    beta1 : float
        First moment decay. Default: 0.9.
    beta2 : float
        Second moment decay. Default: 0.999.
    epsilon : float
        Numerical stability. Default: 1e-8.
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        max_iterations: int = 100,
        convergence_threshold: float = 0.1,
        n_shots: int = 8192,
        verbose: bool = False,
    ):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.max_iter = max_iterations
        self.conv_threshold = convergence_threshold
        self.n_shots = n_shots
        self.verbose = verbose

    def optimize(
        self,
        theta_init: np.ndarray,
        cost_fn: BaseCostFunction,
        circuit: QuantumCircuit,
        noise_model: Optional[NoiseModel] = None,
        seed: Optional[int] = None,
    ) -> OptimizationResult:
        """Run Adam optimization. Returns OptimizationResult."""
        theta = theta_init.copy()
        m = np.zeros_like(theta)   # First moment
        v = np.zeros_like(theta)   # Second moment

        cost_history = []
        grad_norm_history = []
        t_start = time.time()
        converged = False
        final_cost = None

        for iteration in range(self.max_iter):
            current_cost = cost_fn.evaluate(theta, circuit, noise_model, self.n_shots)
            cost_history.append(current_cost)

            if self.verbose:
                print(f"  [Adam] Iter {iteration:3d}  cost={current_cost:+.4f}")

            if abs(current_cost - cost_fn.optimal_value) < self.conv_threshold:
                converged = True
                final_cost = current_cost
                break

            gradient = cost_fn.compute_gradient(theta, circuit, noise_model, self.n_shots)
            grad_norm_history.append(float(np.linalg.norm(gradient)))

            # Adam update
            t = iteration + 1
            m = self.beta1 * m + (1 - self.beta1) * gradient
            v = self.beta2 * v + (1 - self.beta2) * gradient ** 2
            m_hat = m / (1 - self.beta1 ** t)
            v_hat = v / (1 - self.beta2 ** t)
            theta = theta - self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

        else:
            final_cost = cost_fn.evaluate(theta, circuit, noise_model, self.n_shots)
            cost_history.append(final_cost)

        if final_cost is None:
            final_cost = current_cost

        return OptimizationResult(
            method="Adam",
            converged=converged,
            iterations=len(cost_history) - 1,
            final_cost=final_cost,
            wall_clock_time=time.time() - t_start,
            cost_history=cost_history,
            gradient_norm_history=grad_norm_history,
            qfim_condition_history=[],
            theta_final=theta,
            seed=seed,
        )


if __name__ == "__main__":
    import numpy as np
    from src.ansatz import build_ansatz
    from src.cost_function import MaxCutCostFunction
    from src.noise_model import build_noise_model

    print("Vanilla GD Optimizer — Single Trial Demo")
    print("=" * 50)

    np.random.seed(42)
    n_qubits, n_layers = 4, 3
    n_params = n_qubits * 2 * n_layers

    circuit = build_ansatz(n_qubits=n_qubits, n_layers=n_layers)
    cost_fn = MaxCutCostFunction(n_qubits=n_qubits)
    noise_model = build_noise_model("trapped_ion")

    theta_init = np.random.uniform(-0.1, 0.1, n_params)

    optimizer = VanillaGradientDescent(
        learning_rate=0.1,
        max_iterations=20,
        n_shots=1024,
        verbose=True,
    )
    result = optimizer.optimize(theta_init, cost_fn, circuit, noise_model, seed=42)

    print(f"\nResult:")
    print(f"  Converged  : {result.converged}")
    print(f"  Iterations : {result.iterations}")
    print(f"  Final cost : {result.final_cost:.4f}")
    print(f"  Time       : {result.wall_clock_time:.2f}s")
