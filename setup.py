#!/usr/bin/env python
from setuptools import setup, find_packages

from atomic_images import __author__, __version__, __email__, __program__, __description__

setup(
    name=__program__,
    version=__version__,
    packages=find_packages(),
    author=__author__,
    author_email=__email__,
    description=__description__
)
