#!/usr/bin/env python3
from setuptools import find_packages, setup

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
        "torch>=1.13.1"         # Stable PyTorch version
    ],
    extras_require={
        "dev": ["flake8==4.0.1", "isort==5.14.0", "black==22.3.0", "pre-commit==2.25.1"]
    },
)
