import os
import re
import sys

from setuptools import find_packages, setup

deps = [
    "audiomentations", "click", "datasets", "librosa", "matplotlib",
    "monotonic_align", "onnx", "onnxruntime", "phonemizer",
    "pronouncing", "psutil", "pyav", "requests", "safetensors",
    "soundfile", "speechbrain", "tokenizers", "torch", "torch_audiomentations",
    "torchaudio", "torchmetrics", "tqdm", "wandb"
]

extra_deps = {
    "gpu": ["onnxruntime-gpu"],
    "dark": ["qbstyles"]
}

setup(
    name="heybuddy",
    description="Hey Buddy is a tool for training wake-word-detecting neural networks for use in web browsers.",
    version="0.1.0",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    license="Apache 2.0",
    author="Benjamin Paine",
    author_email="painebenjamin@gmail.com",
    url="https://github.com/painebenjamin/heybuddy",
    package_dir={"": "src/python"},
    packages=find_packages("src/python"),
    package_data={"heybuddy": ["py.typed"]},
    include_package_data=True,
    python_requires=">=3.8.0",
    install_requires=deps,
    extras_require=extra_deps,
    entry_points={
        "console_scripts": [
            "heybuddy = heybuddy.__main__:main"
        ]
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Environment :: GPU :: NVIDIA CUDA",
        "Operating System :: POSIX :: Linux",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
    ]
    + [f"Programming Language :: Python :: 3.{i}" for i in range(8, 13)],
)
