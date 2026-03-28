# Quantum Natural Gradient Optimization for NISQ Variational Quantum Algorithms

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Qiskit](https://img.shields.io/badge/Qiskit-1.0+-6929C4.svg)](https://qiskit.org/)
[![PennyLane](https://img.shields.io/badge/PennyLane-0.35+-orange.svg)](https://pennylane.ai/)

> **Paper:** *Quantum Natural Gradient Optimization for Convergence Reliability in NISQ Variational Quantum Algorithms*
> **Author:** Mezbah Uddin Rafi, Council for International Artificial Intelligence Union, Geneva

---

## Overview

Variational Quantum Algorithms (VQAs) are the leading approach for extracting useful computation from Noisy Intermediate-Scale Quantum (NISQ) devices. However, they suffer from a critical failure mode: the **Barren Plateau** phenomenon.

### The Problem: Barren Plateaus

When gradients vanish exponentially with system size, standard gradient descent fails catastrophically:

```
Var[∂C/∂θ] ≤ 2 / [M × 2^(4L)]
```

For a 10-qubit system at depth L=10, gradients become ~10¹² times smaller than for shallow circuits. Vanilla gradient descent needs ~10¹² circuit evaluations to measure a useful gradient — equivalent to ~3,000 years on current hardware.

### The Solution: Quantum Natural Gradient

Quantum Natural Gradient (QNG) preconditions parameter updates using the **Quantum Fisher Information Matrix (QFIM)** — the natural metric tensor on quantum state space:

```
θ(k+1) = θ(k) - η × QFIM(k)⁻¹ × ∇L(θ(k))
```

QFIM automatically amplifies gradient signals in *trainable* directions and suppresses movement in barren plateau regions. This geometric awareness makes QNG dramatically more robust than vanilla gradient descent under realistic NISQ noise.

### Key Results (from the paper)

| Metric | Vanilla GD | QNG | Improvement |
|--------|-----------|-----|-------------|
| Convergence Success Rate | 30% | **95%** | **3.2×** |
| Iterations to Convergence | 95 | **15** | **6.3×** |
| Wall-clock Time | 342 s | **287 s** | **16% faster** |
| Final Cost (mean) | -1.8 | **-3.1** | **72% better** |

*Results on 4-qubit MaxCut with trapped-ion noise model (p₂q = 3.5×10⁻³), 50 independent trials.*

---

## Repository Structure

```
qng-optimizer/
├── src/                        # Core library
│   ├── ansatz.py               # Hardware-efficient ansatz construction
│   ├── cost_function.py        # MaxCut Hamiltonian & cost evaluation
│   ├── qfim.py                 # Quantum Fisher Information Matrix computation
│   ├── qng_optimizer.py        # QNG optimizer (QFIM + regularization)
│   ├── vanilla_gd.py           # Vanilla gradient descent baseline
│   └── noise_model.py          # NISQ noise models (depolarizing + readout)
├── experiments/
│   ├── run_experiments.py      # Full 50-trial experiment suite
│   └── compare_qng_vs_gd.py    # Head-to-head comparison with plots
├── results/
│   ├── sample_output.json      # Pre-computed sample results
│   └── plots.py                # Visualization utilities
├── docs/
│   ├── theory.md               # Mathematical background (barren plateaus, QNG)
│   └── methodology.md          # Experimental methodology details
├── notebooks/
│   └── demo.ipynb              # Interactive walkthrough
├── requirements.txt
├── setup.py
├── .gitignore
└── LICENSE
```

---

## Installation

### Prerequisites

- Python 3.9 or higher
- pip package manager

### Option 1: Install from source (recommended)

```bash
git clone https://github.com/your-username/qng-optimizer.git
cd qng-optimizer
pip install -e .
```

### Option 2: Install dependencies directly

```bash
git clone https://github.com/your-username/qng-optimizer.git
cd qng-optimizer
pip install -r requirements.txt
```

### Verify installation

```bash
python -c "import qiskit, pennylane; print('Installation successful')"
```

---

## Quick Start

### Run a single QNG optimization

```python
from src.ansatz import build_ansatz
from src.cost_function import MaxCutCostFunction
from src.noise_model import build_noise_model
from src.qng_optimizer import QNGOptimizer
import numpy as np

# Define the 4-qubit MaxCut problem (as in the paper)
edge_weights = {(0,1): 1.5, (0,2): 1.0, (1,3): 0.8, (2,3): 1.2}
cost_fn = MaxCutCostFunction(n_qubits=4, edge_weights=edge_weights)

# Build hardware-efficient ansatz (3 layers)
ansatz = build_ansatz(n_qubits=4, n_layers=3)

# Use trapped-ion noise model
noise_model = build_noise_model(platform="trapped_ion")

# Initialize optimizer
optimizer = QNGOptimizer(
    learning_rate=0.1,
    regularization=1e-4,
    max_iterations=100,
    convergence_threshold=0.1
)

# Run optimization
theta_init = np.random.uniform(-0.1, 0.1, 24)  # 4 qubits × 2 rotations × 3 layers
result = optimizer.optimize(theta_init, cost_fn, ansatz, noise_model)

print(f"Converged: {result['converged']}")
print(f"Iterations: {result['iterations']}")
print(f"Final cost: {result['final_cost']:.4f}")
```

### Run full experiment suite

```bash
python experiments/run_experiments.py --n_trials 50 --platform trapped_ion
```

### Compare QNG vs Vanilla GD

```bash
python experiments/compare_qng_vs_gd.py --n_trials 50 --plot
```

---

## Running Experiments

### Full reproduction of paper results

```bash
# Run all 50 trials across all noise models
python experiments/run_experiments.py \
    --n_trials 50 \
    --platforms trapped_ion superconducting high_noise \
    --output_dir results/ \
    --seed 42
```

This will produce:
- `results/experiment_results.json` — raw data for all trials
- `results/convergence_plot.png` — convergence curves
- `results/summary_table.csv` — aggregate statistics

### Quick test (5 trials)

```bash
python experiments/run_experiments.py --n_trials 5 --platform trapped_ion --verbose
```

### Interactive notebook

```bash
jupyter notebook notebooks/demo.ipynb
```

---

## Example Results

After running experiments, you should see output similar to:

```
================================================================================
QNG vs Vanilla GD: Experiment Summary (50 trials, trapped_ion noise)
================================================================================
Method              Success Rate    Mean Iters    Mean Time(s)    Final Cost
--------------------------------------------------------------------------------
Vanilla GD          30.0%           95.0 ± 0.0    342.1 ± 45.2    -1.82 ± 0.41
QNG                 95.0%           15.3 ± 3.1    287.4 ± 22.8    -3.10 ± 0.18
--------------------------------------------------------------------------------
Improvement         3.17×           6.20×         16.0% faster    72% better
================================================================================
```

---

## Understanding the Code

### Core Concepts

**1. Ansatz (`src/ansatz.py`)**
Implements the 3-layer hardware-efficient ansatz:
- Layer structure: `RY(θ) + RZ(θ)` on each qubit, followed by CNOT entangling gates
- Total parameters: `n_qubits × 2 × n_layers = 24` for the default 4-qubit, 3-layer setup

**2. QFIM Computation (`src/qfim.py`)**
Full QFIM using the parameter-shift rule:
- Requires `2P(P+1)` circuit evaluations where P = number of parameters
- Matrix inversion with adaptive regularization: `QFIM_reg = QFIM + λI`

**3. QNG Optimizer (`src/qng_optimizer.py`)**
- Computes gradient via parameter-shift rule
- Computes QFIM, inverts with regularization
- Updates: `θ ← θ - η × QFIM⁻¹ × ∇L`

**4. Noise Models (`src/noise_model.py`)**
Three platforms from the paper:

| Platform | p₁q | p₂q |
|----------|-----|-----|
| `trapped_ion` | 1.0×10⁻⁴ | 3.5×10⁻³ |
| `superconducting` | 1.8×10⁻⁴ | 2.0×10⁻² |
| `high_noise` | 5.0×10⁻⁴ | 1.0×10⁻² |

---

## Hardware Validation Roadmap

As described in the paper, the 24-month path to hardware validation:

| Phase | Timeline | Platform | Action |
|-------|----------|----------|--------|
| **Phase 1** | Months 1–3 | IonQ Forte (cloud) | Validate 4-qubit MaxCut on real hardware (~$5,000) |
| **Phase 2** | Months 4–12 | Quantinuum Helios | Partner access, 8–12 qubit experiments |
| **Phase 3** | Months 13–24 | Radiation lab | Space qualification under cosmic radiation conditions |

---

## Extending the Work

### Add a new optimizer

```python
# src/my_optimizer.py
from src.cost_function import MaxCutCostFunction

class MyOptimizer:
    def __init__(self, learning_rate=0.1, max_iterations=100):
        self.lr = learning_rate
        self.max_iter = max_iterations

    def optimize(self, theta_init, cost_fn, ansatz, noise_model):
        # Your optimizer logic here
        ...
        return {"converged": ..., "iterations": ..., "final_cost": ...}
```

### Add a new problem

```python
# Implement a new cost function
from src.cost_function import BaseCostFunction

class MyProblem(BaseCostFunction):
    def evaluate(self, theta, ansatz, noise_model):
        # Compute and return cost
        ...

    def get_hamiltonian(self):
        # Return SparsePauliOp
        ...
```

### Implement efficient QFIM (future work)

The paper identifies O(1) QFIM estimation (Stein's Identity, Kolotouros & Wallden 2024) as the top future work priority. See `docs/theory.md` for the mathematical framework.

---

## Citation

If you use this code in your research, please cite the original paper:

```bibtex
@article{rafi2025qng,
  title   = {Quantum Natural Gradient Optimization for Convergence Reliability
             in NISQ Variational Quantum Algorithms},
  author  = {Rafi, Mezbah Uddin},
  journal = {Council for International Artificial Intelligence Union},
  year    = {2025},
  address = {Geneva, Switzerland}
}
```

---

## References

Key references from the paper:

- McClean et al. (2018) — Barren plateaus in quantum neural networks. *Nature Communications*, 9(4812).
- Stokes et al. (2020) — Quantum natural gradient. *Quantum*, 4, 269.
- Cerezo et al. (2021) — Cost function dependent barren plateaus. *Nature Communications*, 12, 1791.
- Wang et al. (2021) — Noise-induced barren plateaus. *Nature Communications*, 12, 6961.
- Kolotouros & Wallden (2024) — Random natural gradient. *Quantum Journal*, 8(1), 1478.
- Minervini et al. (2025) — QFIM estimation via Stein's identity. arXiv:2502.17231.

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Acknowledgments

We acknowledge the Qiskit development team and PennyLane framework creators for open-source quantum computing tools. Computational resources were provided by the Council for International Artificial Intelligence Union, Geneva.
