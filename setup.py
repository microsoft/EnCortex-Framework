from os import path

import encortex
from setuptools import find_packages, setup


with open("requirements.txt") as f:
    required = f.read().splitlines()

curdir = path.abspath(path.dirname(__file__))
with open(path.join(curdir, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="encortex",
    packages=find_packages(),
    version=encortex.__version__,
    license="Early Access Program",
    description="EnCortex",
    long_description=long_description,
    long_description_content_type="text/markdown",
    entry_points={
        "console_scripts": [
            "encortex_setup = encortex.__init__:setup_encortex",
            "encortex_aml = encortex.azure:main",
            "encortex_azure = encortex.azure_utils:azure_app",
        ]
    },
    author="Vaibhav Balloli, Millend Roy, Anupam Sobti, Apoorva Agrawal, Akshay Nambi, Tanuja Ganu",
    author_email="t-vballoli@microsoft.com",
    url="https://dev.azure.com/MSREnergy/_git/EnCortex-Release",
    keywords=[
        "energy market",
        "reinforcement learning",
        "mixed integer linear programming",
        "environment",
        "azure",
        "azure machine learning",
    ],
    install_requires=required,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: CELA Approved :: EAP License",
        "Programming Language :: Python :: 3.8",
    ],
)
