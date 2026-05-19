from setuptools import setup, find_packages

setup(
    name="thesis-vcc-gnn",
    version="1.0.0",
    description="Heterogeneous GNN for commit-level vulnerability detection (VCC prediction)",
    author="Michel Moussally",
    packages=find_packages(include=["src", "src.*"]),
    python_requires=">=3.11",
    install_requires=[
        "torch>=2.6.0",
        "torch_geometric>=2.6.0",
        "numpy>=1.24",
        "pandas>=1.5",
        "scikit-learn>=1.2",
        "scipy>=1.10",
        "tqdm>=4.64",
        "matplotlib>=3.6",
        "networkx>=3.0",
    ],
)
