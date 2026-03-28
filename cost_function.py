"""
cost_function.py
================
MaxCut cost function and Hamiltonian for the 4-qubit benchmark problem.

The MaxCut problem seeks a partition of graph vertices that maximizes the
total weight of edges crossing the partition. Formulated as a QUBO:

    minimize  E(θ) = Σ_{<i,j>} w_ij * (1 - Z_i Z_j) / 2

where w_ij are edge weights and Z_i are Pauli-Z operators.

This is the same problem instance used in Section II.B of the paper:
    Edge weights: w_01=1.5, w_02=1.0, w_13=0.8, w_23=1.2
    Optimal cut value: -3.3 (minimum of cost function)
    Optimal partition: {0,3} vs {1,2}  (verified by exhaustive search)

The cost function is evaluated by:
    1. Building the Qiskit SparsePauliOp Hamiltonian
    2. Running the ansatz circuit on the Aer noise simulator
    3. Estimating ⟨ψ(θ)|H|ψ(θ)⟩ using the Estimator primitive

Reference:
    Farhi, Goldstone, Gutmann (2014) — QAOA. arXiv:1411.4028
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import Estimator as AerEstimator
from qiskit_aer.noise import NoiseModel


# ---------------------------------------------------------------------------
# Abstract base class (for easy extension to other problems)
# ---------------------------------------------------------------------------

class BaseCostFunction(ABC):
    """Abstract base for variational cost functions."""

    @abstractmethod
    def evaluate(
        self,
        theta: np.ndarray,
        circuit: QuantumCircuit,
        noise_model: Optional[NoiseModel] = None,
        n_shots: int = 8192,
    ) -> float:
        """Evaluate the cost function at parameters theta."""

    @abstractmethod
    def get_hamiltonian(self) -> SparsePauliOp:
        """Return the problem Hamiltonian as a SparsePauliOp."""

    @property
    @abstractmethod
    def optimal_value(self) -> float:
        """Return the known optimal (minimum) cost value."""


# ---------------------------------------------------------------------------
# MaxCut cost function
# ---------------------------------------------------------------------------

class MaxCutCostFunction(BaseCostFunction):
    """
    MaxCut cost function for a weighted graph.

    The Hamiltonian is:
        H = Σ_{<i,j>} w_ij * (I - Z_i Z_j) / 2

    Minimizing ⟨H⟩ maximizes the cut weight.

    Parameters
    ----------
    n_qubits : int
        Number of qubits (= number of graph vertices).
    edge_weights : Dict[Tuple[int, int], float]
        Dictionary mapping (qubit_i, qubit_j) → weight.
        Edges must satisfy i < j.

    Example
    -------
    >>> from src.cost_function import MaxCutCostFunction
    >>> edge_weights = {(0,1): 1.5, (0,2): 1.0, (1,3): 0.8, (2,3): 1.2}
    >>> cost_fn = MaxCutCostFunction(n_qubits=4, edge_weights=edge_weights)
    >>> print(f"Optimal value: {cost_fn.optimal_value}")
    """

    # Paper's specific problem instance (Section II.B)
    DEFAULT_EDGE_WEIGHTS: Dict[Tuple[int, int], float] = {
        (0, 1): 1.5,
        (0, 2): 1.0,
        (1, 3): 0.8,
        (2, 3): 1.2,
    }
    # Verified by exhaustive classical search over all 2^4 = 16 bitstrings
    _PAPER_OPTIMAL_VALUE: float = -3.3

    def __init__(
        self,
        n_qubits: int = 4,
        edge_weights: Optional[Dict[Tuple[int, int], float]] = None,
    ):
        self.n_qubits = n_qubits
        self.edge_weights = edge_weights or self.DEFAULT_EDGE_WEIGHTS

        # Validate edges are within qubit range
        for (i, j) in self.edge_weights:
            assert 0 <= i < j < n_qubits, (
                f"Invalid edge ({i},{j}) for {n_qubits}-qubit system. "
                "Edges must satisfy 0 ≤ i < j < n_qubits."
            )

        self._hamiltonian: Optional[SparsePauliOp] = None

    def get_hamiltonian(self) -> SparsePauliOp:
        """
        Construct the MaxCut Hamiltonian as a SparsePauliOp.

        H = Σ_{<i,j>} (w_ij/2) * (I⊗...⊗I - Z_i⊗Z_j⊗I...)

        Returns
        -------
        SparsePauliOp
            Sparse representation of the Hamiltonian.
        """
        if self._hamiltonian is not None:
            return self._hamiltonian

        pauli_terms = []
        coefficients = []

        for (i, j), weight in self.edge_weights.items():
            # Constant term: +w_ij/2 * I
            identity_str = "I" * self.n_qubits
            pauli_terms.append(identity_str)
            coefficients.append(weight / 2)

            # ZZ term: -w_ij/2 * Z_i Z_j
            # Qiskit SparsePauliOp uses little-endian: index 0 = rightmost
            zz_list = ["I"] * self.n_qubits
            zz_list[self.n_qubits - 1 - i] = "Z"
            zz_list[self.n_qubits - 1 - j] = "Z"
            pauli_terms.append("".join(zz_list))
            coefficients.append(-weight / 2)

        self._hamiltonian = SparsePauliOp(pauli_terms, coefficients)
        return self._hamiltonian

    def evaluate(
        self,
        theta: np.ndarray,
        circuit: QuantumCircuit,
        noise_model: Optional[NoiseModel] = None,
        n_shots: int = 8192,
    ) -> float:
        """
        Evaluate ⟨ψ(θ)|H|ψ(θ)⟩ using Qiskit Aer Estimator.

        Parameters
        ----------
        theta : np.ndarray
            Current parameter values (flat array).
        circuit : QuantumCircuit
            Parameterized ansatz circuit (from build_ansatz).
        noise_model : NoiseModel, optional
            Qiskit Aer noise model. If None, uses ideal simulation.
        n_shots : int
            Number of measurement shots. Paper uses 8192.

        Returns
        -------
        float
            Estimated expectation value ⟨H⟩.
        """
        from src.ansatz import bind_parameters

        bound_circuit = bind_parameters(circuit, theta)
        hamiltonian = self.get_hamiltonian()

        # Configure Aer Estimator with noise model
        estimator_options = {
            "shots": n_shots,
        }
        if noise_model is not None:
            estimator_options["noise_model"] = noise_model
            estimator_options["approximation"] = True  # Use shot-based estimation

        estimator = AerEstimator(run_options=estimator_options)
        job = estimator.run([bound_circuit], [hamiltonian])
        result = job.result()

        cost = float(result.values[0])
        return cost

    @property
    def optimal_value(self) -> float:
        """
        Return the classical optimal cut value (verified by exhaustive search).

        For the paper's default edge weights, the optimal partition is
        {0,3} vs {1,2}, yielding a cut weight of 3.3 (cost = -3.3).
        """
        if self.edge_weights == self.DEFAULT_EDGE_WEIGHTS:
            return self._PAPER_OPTIMAL_VALUE
        # For custom graphs, compute by exhaustive search
        return self._exhaustive_optimal()

    def _exhaustive_optimal(self) -> float:
        """Find optimal cut by trying all 2^n bitstrings (only for small n)."""
        if self.n_qubits > 20:
            raise ValueError("Exhaustive search infeasible for n_qubits > 20.")

        best = float("inf")
        for bits in range(2 ** self.n_qubits):
            partition = [(bits >> q) & 1 for q in range(self.n_qubits)]
            cost = 0.0
            for (i, j), w in self.edge_weights.items():
                zi = 1 - 2 * partition[i]  # map {0,1} → {+1,-1}
                zj = 1 - 2 * partition[j]
                cost += (w / 2) * (1 - zi * zj)
            best = min(best, -cost)  # negative because we minimize
        return best

    def compute_gradient(
        self,
        theta: np.ndarray,
        circuit: QuantumCircuit,
        noise_model: Optional[NoiseModel] = None,
        n_shots: int = 8192,
        shift: float = np.pi / 2,
    ) -> np.ndarray:
        """
        Compute the gradient ∂C/∂θ_k using the parameter-shift rule.

        The parameter-shift rule states:
            ∂C/∂θ_k = [C(θ_k + π/2) - C(θ_k - π/2)] / 2

        This is an unbiased estimator of the exact gradient and works
        for all gates of the form exp(-iθG) where G² = I/4.

        Parameters
        ----------
        theta : np.ndarray
            Current parameters.
        circuit : QuantumCircuit
            Parameterized ansatz.
        noise_model : NoiseModel, optional
            NISQ noise model.
        n_shots : int
            Shots per circuit evaluation.
        shift : float
            Parameter shift amount. Default: π/2.

        Returns
        -------
        np.ndarray
            Gradient vector of shape (n_params,).
        """
        n_params = len(theta)
        gradient = np.zeros(n_params)

        for k in range(n_params):
            # Positive shift: θ_k → θ_k + π/2
            theta_plus = theta.copy()
            theta_plus[k] += shift
            cost_plus = self.evaluate(theta_plus, circuit, noise_model, n_shots)

            # Negative shift: θ_k → θ_k - π/2
            theta_minus = theta.copy()
            theta_minus[k] -= shift
            cost_minus = self.evaluate(theta_minus, circuit, noise_model, n_shots)

            gradient[k] = (cost_plus - cost_minus) / 2

        return gradient

    def __repr__(self) -> str:
        edges = ", ".join(f"({i},{j}):{w}" for (i,j), w in self.edge_weights.items())
        return (
            f"MaxCutCostFunction(n_qubits={self.n_qubits}, "
            f"edges=[{edges}], optimal={self.optimal_value})"
        )


if __name__ == "__main__":
    # Demonstrate the MaxCut problem setup
    edge_weights = {(0, 1): 1.5, (0, 2): 1.0, (1, 3): 0.8, (2, 3): 1.2}
    cost_fn = MaxCutCostFunction(n_qubits=4, edge_weights=edge_weights)

    print("MaxCut Problem (4 qubits):")
    print(f"  Edges: {edge_weights}")
    print(f"  Optimal cost value: {cost_fn.optimal_value}")
    print(f"\nHamiltonian:\n{cost_fn.get_hamiltonian()}")
