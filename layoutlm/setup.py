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
        "transformers==4.27.2",  # Latest stable version
        "tensorboardX==2.5.1",   # Latest stable version
        "lxml==4.9.1",          # Latest stable version
        "seqeval==1.2.1",        # Latest stable version
        "Pillow==9.5.0",        # Latest stable version
    ],
    extras_require={
        "dev": ["flake8==4.0.1", "isort==5.14.0", "black==22.3.0", "pre-commit==2.25.1"]
    },
)
