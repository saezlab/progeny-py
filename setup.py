#!/usr/bin/env python

from distutils.core import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='progeny-py',
    version='1.0.2',
    author='Pau Badia i Mompel',
    author_email="pau.badia@uni-heidelberg.de",
    description='progeny-py is a python package to compute pathway activity \
    from RNA-seq data using PROGENy',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/saezlab/progeny-py',
    project_urls={
        "Bug Tracker": "https://github.com/saezlab/progeny-py/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    packages=['progeny'], 
    license='LICENSE.txt',
    package_data={
       'progeny': ['data/model_human_full.pkl', 
                   'data/model_mouse_full.pkl']
    },
    install_requires=[
        'anndata',
        'scanpy',
        'numpy',
        'pandas',
        'tqdm'
    ]
)
