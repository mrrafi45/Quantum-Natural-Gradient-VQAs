"""
noise_model.py
==============
NISQ noise model construction using Qiskit Aer.

Implements three hardware platforms from the paper (Table 1):
  - trapped_ion   : IonQ Forte   (p1q=1e-4, p2q=3.5e-3)
  - superconducting: IBM Falcon   (p1q=1.8e-4, p2q=2e-2)
  - high_noise    : Degraded HW  (p1q=5e-4, p2q=1e-2)

Noise channels modeled:
  - Depolarizing error on single-qubit gates
  - Depolarizing error on two-qubit (CNOT) gates
  - Readout (measurement) error
  - Thermal relaxation (T1/T2 decoherence)

Reference:
    Wang et al. (2021) — Noise-induced barren plateaus. Nat Commun 12, 6961.
    Qiskit Aer documentation (https://qiskit.github.io/qiskit-aer/)
"""

from dataclasses import dataclass
from typing import Optional

from qiskit_aer.noise import (
    NoiseModel,
    depolarizing_error,
    ReadoutError,
    thermal_relaxation_error,
)
import numpy as np


# ---------------------------------------------------------------------------
# Hardware platform specifications (Table 1 from the paper)
# ---------------------------------------------------------------------------

PLATFORM_SPECS = {
    "trapped_ion": {
        "description": "IonQ Forte — 32-qubit trapped-ion (Nov 2025)",
        "p1q": 1.0e-4,          # Single-qubit gate error rate
        "p2q": 3.5e-3,          # Two-qubit gate error rate
        "readout_error": 0.01,  # 1% readout error
        "t1_ms": 100.0,         # Amplitude damping time (ms)
        "t2_ms": 50.0,          # Dephasing time (ms)
        "gate_time_ns": 100.0,  # Gate time (ns)
    },
    "superconducting": {
        "description": "IBM Falcon — 27-qubit superconducting (Nov 2025)",
        "p1q": 1.8e-4,
        "p2q": 2.0e-2,
        "readout_error": 0.02,
        "t1_ms": 10.0,
        "t2_ms": 5.0,
        "gate_time_ns": 50.0,
    },
    "high_noise": {
        "description": "Degraded hardware / older NISQ systems",
        "p1q": 5.0e-4,
        "p2q": 1.0e-2,
        "readout_error": 0.03,
        "t1_ms": 20.0,
        "t2_ms": 10.0,
        "gate_time_ns": 200.0,
    },
    "noiseless": {
        "description": "Ideal noiseless simulation (for debugging)",
        "p1q": 0.0,
        "p2q": 0.0,
        "readout_error": 0.0,
        "t1_ms": 1e9,
        "t2_ms": 1e9,
        "gate_time_ns": 100.0,
    },
}


@dataclass
class NoiseSpec:
    """Holds the numerical noise parameters for one hardware platform."""
    platform: str
    description: str
    p1q: float           # Single-qubit depolarizing error probability
    p2q: float           # Two-qubit depolarizing error probability
    readout_error: float # Measurement readout error probability
    t1_ms: float         # T1 relaxation time in milliseconds
    t2_ms: float         # T2 dephasing time in milliseconds
    gate_time_ns: float  # Gate execution time in nanoseconds


def get_noise_spec(platform: str) -> NoiseSpec:
    """
    Return the NoiseSpec for a given platform name.

    Parameters
    ----------
    platform : str
        One of 'trapped_ion', 'superconducting', 'high_noise', 'noiseless'.

    Returns
    -------
    NoiseSpec
    """
    if platform not in PLATFORM_SPECS:
        raise ValueError(
            f"Unknown platform '{platform}'. "
            f"Choose from: {list(PLATFORM_SPECS.keys())}"
        )
    spec = PLATFORM_SPECS[platform]
    return NoiseSpec(platform=platform, **spec)


def build_noise_model(
    platform: str = "trapped_ion",
    n_qubits: Optional[int] = None,
    include_thermal: bool = True,
) -> NoiseModel:
    """
    Construct a Qiskit Aer NoiseModel for the specified hardware platform.

    The noise model includes:
      1. Depolarizing error on all single-qubit gates (u, u1, u2, u3, rx, ry, rz)
      2. Depolarizing error on all two-qubit gates (cx, cz, ecr)
      3. Readout (measurement) error on all qubits
      4. Thermal relaxation (T1/T2) on all gates (if include_thermal=True)

    Parameters
    ----------
    platform : str
        Hardware platform. Options: 'trapped_ion', 'superconducting',
        'high_noise', 'noiseless'. Default: 'trapped_ion'.
    n_qubits : int, optional
        Number of qubits (needed for readout error matrix). If None, a
        generic single-qubit readout error is used.
    include_thermal : bool
        Whether to include thermal relaxation (T1/T2) errors. Default True.

    Returns
    -------
    qiskit_aer.noise.NoiseModel
        Ready-to-use noise model for Aer simulator.

    Example
    -------
    >>> from src.noise_model import build_noise_model
    >>> nm = build_noise_model("trapped_ion")
    >>> print(nm)
    """
    spec = get_noise_spec(platform)
    noise_model = NoiseModel()

    # -----------------------------------------------------------------------
    # 1. Single-qubit gate depolarizing error
    # -----------------------------------------------------------------------
    if spec.p1q > 0:
        error_1q = depolarizing_error(spec.p1q, 1)
        # Apply to standard single-qubit rotation gates used in the ansatz
        single_qubit_gates = ["u", "u1", "u2", "u3", "rx", "ry", "rz", "h", "x", "y", "z"]
        noise_model.add_all_qubit_quantum_error(error_1q, single_qubit_gates)

    # -----------------------------------------------------------------------
    # 2. Two-qubit gate depolarizing error
    # -----------------------------------------------------------------------
    if spec.p2q > 0:
        error_2q = depolarizing_error(spec.p2q, 2)
        two_qubit_gates = ["cx", "cz", "ecr", "swap"]
        noise_model.add_all_qubit_quantum_error(error_2q, two_qubit_gates)

    # -----------------------------------------------------------------------
    # 3. Readout (measurement) error
    # -----------------------------------------------------------------------
    if spec.readout_error > 0:
        p_meas = spec.readout_error
        # Symmetric readout error: P(0|1) = P(1|0) = p_meas
        readout_error = ReadoutError([
            [1 - p_meas, p_meas],   # P(outcome|state=0): [P(0|0), P(1|0)]
            [p_meas, 1 - p_meas],   # P(outcome|state=1): [P(0|1), P(1|1)]
        ])
        noise_model.add_all_qubit_readout_error(readout_error)

    # -----------------------------------------------------------------------
    # 4. Thermal relaxation (T1/T2 decoherence)
    # -----------------------------------------------------------------------
    if include_thermal and spec.t1_ms < 1e8:
        t1_ns = spec.t1_ms * 1e6   # Convert ms → ns
        t2_ns = spec.t2_ms * 1e6
        gate_time = spec.gate_time_ns

        # Ensure T2 ≤ 2*T1 (physical constraint)
        t2_ns = min(t2_ns, 2 * t1_ns)

        # Single-qubit thermal relaxation
        thermal_1q = thermal_relaxation_error(t1_ns, t2_ns, gate_time)
        noise_model.add_all_qubit_quantum_error(
            thermal_1q, ["u", "u1", "u2", "u3", "rx", "ry", "rz"]
        )

        # Two-qubit thermal relaxation (applied per qubit, sequential)
        thermal_2q = thermal_relaxation_error(t1_ns, t2_ns, 2 * gate_time)
        thermal_2q_tensor = thermal_2q.expand(thermal_2q)
        noise_model.add_all_qubit_quantum_error(thermal_2q_tensor, ["cx", "cz"])

    return noise_model


def print_noise_summary(platform: str) -> None:
    """Print a human-readable summary of a platform's noise parameters."""
    spec = get_noise_spec(platform)
    print(f"\nNoise Model: {spec.description}")
    print(f"  Single-qubit gate error : {spec.p1q:.2e}")
    print(f"  Two-qubit gate error    : {spec.p2q:.2e}")
    print(f"  Readout error           : {spec.readout_error:.2%}")
    print(f"  T1 relaxation           : {spec.t1_ms} ms")
    print(f"  T2 dephasing            : {spec.t2_ms} ms")
    print(f"  Gate time               : {spec.gate_time_ns} ns")


if __name__ == "__main__":
    for platform in PLATFORM_SPECS:
        print_noise_summary(platform)
