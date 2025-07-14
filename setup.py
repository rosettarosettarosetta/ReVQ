# ------------------------------------------------------------------------------
# ReVQ: Quantize-then-Rectify: Efficient VQ-VAE Training
# Copyright (c) 2025 Borui Zhang. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------------

from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="xvq",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=requirements,
    version="0.1.0",
    description="A package for tokenization",
    author="Borui Zhang",
)