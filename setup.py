#!/usr/bin/env python
"""The setup script."""

from setuptools import find_packages, setup

with open("README.md") as readme_file:
    readme = readme_file.read()

with open("HISTORY.md") as history_file:
    history = history_file.read()

# NOTE: Update pinned_reqs whenever install_requires or extras_require changes.
install_requires = [
    "numpy>=1.17.0",  # >=1.17.0 that is when default_rng becomes available.
    "numba>=0.51.0",
    "pandas>=1.0.0",
    "toml>=0.10.0",
    "sortedcontainers>=2.0.0",  # Primarily used in SlidingBoundariesArchive.
    "scikit-learn>=0.20.0",  # Primarily used in CVTArchive.
    "scipy>=1.4.0",  # Primarily used in CVTArchive.
    "decorator>=4.0.0",
    "threadpoolctl>=2.0.0",
]

extras_require = {
    "visualize": ["matplotlib>=3.0.0",],
    # Dependencies for examples (NOT tutorials -- tutorial notebooks should
    # install deps with cell magic and only depend on ribs and ribs[visualize]).
    "examples": [
        "matplotlib>=3.0.0",
        "gym~=0.17.0",  # Strict since different gym may give different results.
        "Box2D~=2.3.10",  # Used in envs such as Lunar Lander.
        "fire>=0.4.0",
        "alive-progress>=1.0.0",

        # Dask
        "dask>=2.0.0",
        "distributed>=2.0.0",
        "bokeh>=2.0.0",  # Dask dashboard.
    ],
    "dev": [
        "pip>=20.3",
        "pylint",
        "yapf",

        # Testing
        "pytest==7.0.1",
        "pytest-cov==3.0.0",
        "pytest-benchmark==3.4.1",
        "pytest-xdist==2.5.0",

        # Documentation
        "Sphinx==3.2.1",
        "sphinx-material==0.0.32",
        "sphinx-autobuild==2020.9.1",
        "sphinx-copybutton==0.3.1",
        "myst-nb==0.10.1",
        "sphinx-toolbox==2.12.1",

        # Distribution
        "bump2version==0.5.11",
        "wheel==0.36.2",
        "twine==1.14.0",
        "check-wheel-contents==0.2.0",
    ]
}

setup(
    author="ICAROS Lab pyribs Team",
    author_email="team@pyribs.org",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    description=
    "A bare-bones Python library for quality diversity optimization.",
    install_requires=install_requires,
    extras_require=extras_require,
    license="MIT license",
    long_description=readme + "\n\n" + history,
    long_description_content_type="text/markdown",
    include_package_data=True,
    keywords="ribs",
    name="ribs",
    packages=find_packages(include=["ribs", "ribs.*"]),
    python_requires=">=3.7.0",
    test_suite="tests",
    url="https://github.com/icaros-usc/pyribs",
    version="0.4.0",
    zip_safe=False,
)
