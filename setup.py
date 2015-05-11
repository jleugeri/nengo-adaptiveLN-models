#!/usr/bin/env python
import imp
import os

try:
    from setuptools import setup
except ImportError:
    from ez_setup import use_setuptools
    use_setuptools()

from setuptools import find_packages, setup

root = os.path.dirname(os.path.realpath(__file__))
description = ("This package extends the NENGO system by several adaptive linear-non-linear neuron models that map normally distributed inputs to various output distributions.")
with open(os.path.join(root, 'README.md')) as readme:
    long_description = readme.read()

setup(
    name="nengo_adaptiveLN_models",
    version=1.0,
    author="Johannes Leugering",
    author_email="jleugeri@uos.de",
    packages=find_packages(),
    include_package_data=True,
    scripts=[],
    url="https://github.com/jleugeri/nengo-adaptiveLN-models.git",
    license="https://github.com/jleugeri/nengo-adaptiveLN-models/blob/master/COPYING",
    description=description,
    install_requires=[
        "nengo", "scipy", "numpy",
    ],
)
