#!/usr/bin/env python3
import torch
from setuptools import find_packages, setup

torch_ver = [int(x) for x in torch.__version__.split(".")[:2]]
assert torch_ver >= [1, 4], "Requires PyTorch >= 1.4"

setup(
    name="layoutlm",
    version="0.0",
    author="Yiheng Xu",
    url="https://github.com/microsoft/unilm/tree/master/layoutlm",
    description="LayoutLM",
    packages=find_packages(exclude=("configs", "tests")),
    python_requires=">=3.6",
    install_requires=[
        "transformers",  # Latest stable version
        "tensorboardX",   # Latest stable version
        "lxml",          # Latest stable version
        "seqeval",        # Latest stable version
        "Pillow",        # Latest stable version
    ],
    extras_require={
        "dev": ["flake8==4.0.1", "isort==5.14.0", "black==22.3.0", "pre-commit==2.25.1"]
    },
)
