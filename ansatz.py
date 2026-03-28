"""
ansatz.py
=========
Hardware-efficient ansatz construction for NISQ variational quantum algorithms.

Implements the 3-layer hardware-efficient ansatz described in Section II.B
of the paper. The structure per layer is:

    Layer k:
        ├── RY(θ_{k,q,0})  for each qubit q     (rotation)
        ├── RZ(θ_{k,q,1})  for each qubit q     (rotation)
        └── CNOT chain     q→q+1 for all q      (entanglement)

Total parameters: n_qubits × 2 × n_layers
For default (n_qubits=4, n_layers=3): 4 × 2 × 3 = 24 parameters.

The ansatz is implemented in both Qiskit (QuantumCircuit) and PennyLane
(qnode-compatible) to match the paper's dual-framework approach.

Reference:
    Leone et al. (2024) — On the practical usefulness of the hardware efficient ansatz.
    Quantum, 8, 1395. https://doi.org/10.22331/q-2024-10-24-1395
"""

from typing import List, Optional, Tuple
import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector


# ---------------------------------------------------------------------------
# Qiskit ansatz (used for circuit simulation and noise model application)
# ---------------------------------------------------------------------------

def build_ansatz(
    n_qubits: int = 4,
    n_layers: int = 3,
    entanglement: str = "linear",
) -> QuantumCircuit:
    """
    Build the hardware-efficient ansatz as a parameterized Qiskit QuantumCircuit.

    Architecture per layer:
        1. RY(θ) on each qubit
        2. RZ(φ) on each qubit
        3. CNOT entangling gates (linear chain by default)

    Parameters
    ----------
    n_qubits : int
        Number of qubits. Paper uses 4. Default: 4.
    n_layers : int
        Number of repeated layers. Paper uses 3. Default: 3.
    entanglement : str
        Entanglement pattern. Options:
          'linear'  — CNOT(q, q+1) for q = 0, ..., n-2   (paper default)
          'full'    — CNOT between all pairs
          'circular'— Linear + CNOT(n-1, 0)

    Returns
    -------
    QuantumCircuit
        Parameterized circuit. Access parameters via `circuit.parameters`.

    Example
    -------
    >>> from src.ansatz import build_ansatz
    >>> qc = build_ansatz(n_qubits=4, n_layers=3)
    >>> print(f"Parameters: {qc.num_parameters}")   # 24
    >>> print(qc.draw())
    """
    n_params = n_qubits * 2 * n_layers
    params = ParameterVector("θ", length=n_params)

    qc = QuantumCircuit(n_qubits)

    # Optional: Initialize in superposition (helps escape local minima)
    # qc.h(range(n_qubits))

    param_idx = 0

    for layer in range(n_layers):
        # ---- Rotation layer: RY + RZ on each qubit ----
        for qubit in range(n_qubits):
            qc.ry(params[param_idx], qubit)
            param_idx += 1
            qc.rz(params[param_idx], qubit)
            param_idx += 1

        # ---- Entangling layer: CNOT gates ----
        _add_entanglement(qc, n_qubits, entanglement)

        # Barrier for visual clarity (optional, does not affect simulation)
        if layer < n_layers - 1:
            qc.barrier()

    return qc


def _add_entanglement(qc: QuantumCircuit, n_qubits: int, pattern: str) -> None:
    """Apply CNOT gates according to the chosen entanglement pattern."""
    if pattern == "linear":
        for q in range(n_qubits - 1):
            qc.cx(q, q + 1)

    elif pattern == "full":
        for control in range(n_qubits):
            for target in range(control + 1, n_qubits):
                qc.cx(control, target)

    elif pattern == "circular":
        for q in range(n_qubits - 1):
            qc.cx(q, q + 1)
        if n_qubits > 1:
            qc.cx(n_qubits - 1, 0)

    else:
        raise ValueError(
            f"Unknown entanglement pattern '{pattern}'. "
            "Choose from: 'linear', 'full', 'circular'."
        )


def bind_parameters(circuit: QuantumCircuit, theta: np.ndarray) -> QuantumCircuit:
    """
    Bind a numpy array of parameter values to a parameterized QuantumCircuit.

    Parameters
    ----------
    circuit : QuantumCircuit
        Parameterized circuit from build_ansatz().
    theta : np.ndarray
        1D array of parameter values. Length must match circuit.num_parameters.

    Returns
    -------
    QuantumCircuit
        Circuit with all parameters bound (ready to run on simulator).
    """
    if len(theta) != circuit.num_parameters:
        raise ValueError(
            f"Parameter count mismatch: circuit has {circuit.num_parameters} "
            f"parameters, but theta has length {len(theta)}."
        )
    param_dict = {
        param: float(val)
        for param, val in zip(circuit.parameters, theta)
    }
    return circuit.assign_parameters(param_dict)


def get_parameter_count(n_qubits: int, n_layers: int) -> int:
    """Return the number of trainable parameters for the given ansatz shape."""
    return n_qubits * 2 * n_layers


def parameter_shapes(n_qubits: int, n_layers: int) -> dict:
    """
    Return metadata about the parameter structure.

    Useful for understanding which index corresponds to which
    (layer, qubit, rotation_type) triple.
    """
    shapes = {}
    idx = 0
    for layer in range(n_layers):
        for qubit in range(n_qubits):
            shapes[idx] = {"layer": layer, "qubit": qubit, "gate": "RY"}
            idx += 1
            shapes[idx] = {"layer": layer, "qubit": qubit, "gate": "RZ"}
            idx += 1
    return shapes


# ---------------------------------------------------------------------------
# PennyLane ansatz (used for automatic differentiation utilities)
# ---------------------------------------------------------------------------

def pennylane_ansatz(theta: np.ndarray, n_qubits: int = 4, n_layers: int = 3) -> None:
    """
    PennyLane-compatible ansatz function (call inside a QNode).

    Parameters
    ----------
    theta : np.ndarray
        Flat array of parameters, length = n_qubits * 2 * n_layers.
    n_qubits : int
        Number of qubits.
    n_layers : int
        Number of layers.

    Usage
    -----
    >>> import pennylane as qml
    >>> dev = qml.device("default.qubit", wires=4)
    >>>
    >>> @qml.qnode(dev)
    ... def circuit(theta):
    ...     pennylane_ansatz(theta, n_qubits=4, n_layers=3)
    ...     return qml.expval(qml.PauliZ(0))
    """
    try:
        import pennylane as qml
    except ImportError:
        raise ImportError("PennyLane is required. Install with: pip install pennylane")

    param_idx = 0
    for layer in range(n_layers):
        for qubit in range(n_qubits):
            qml.RY(theta[param_idx], wires=qubit)
            param_idx += 1
            qml.RZ(theta[param_idx], wires=qubit)
            param_idx += 1

        # Linear CNOT entanglement
        for q in range(n_qubits - 1):
            qml.CNOT(wires=[q, q + 1])


if __name__ == "__main__":
    # Quick sanity check
    qc = build_ansatz(n_qubits=4, n_layers=3)
    print(f"Ansatz built: {qc.num_qubits} qubits, {qc.num_parameters} parameters")
    print(f"Circuit depth: {qc.depth()}")
    print()
    print(qc.draw(output="text", fold=80))
