"""
QNG Optimizer — Quantum Natural Gradient for NISQ VQAs.

Modules:
    ansatz        — Hardware-efficient ansatz construction
    cost_function — MaxCut Hamiltonian and cost evaluation
    qfim          — Quantum Fisher Information Matrix
    qng_optimizer — QNG optimizer with QFIM preconditioning
    vanilla_gd    — Vanilla gradient descent baseline
    noise_model   — NISQ noise models (depolarizing + readout)
"""
