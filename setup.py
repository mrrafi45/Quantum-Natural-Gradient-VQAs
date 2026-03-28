"""
Setup configuration for the QNG Optimizer package.

Quantum Natural Gradient Optimization for Convergence Reliability
in NISQ Variational Quantum Algorithms.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [
        line.strip()
        for line in fh
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="qng-optimizer",
    version="1.0.0",
    author="Mezbah Uddin Rafi",
    author_email="info.m.rafi19@gmail.com",
    description=(
        "Quantum Natural Gradient Optimization for NISQ Variational Quantum Algorithms"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/qng-optimizer",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "qng-run=experiments.run_experiments:main",
            "qng-compare=experiments.compare_qng_vs_gd:main",
        ]
    },
)
