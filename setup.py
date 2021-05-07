#!/usr/bin/env python
import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()
    
setuptools.setup(
    name="progeny-py",
    version="1.0.1",
    author="Pau Badia i Mompel",
    author_email="pau.badia@uni-heidelberg.de",
    description="progeny-py is a python package to compute pathway activity \
    from RNA-seq data using PROGENy",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/saezlab/progeny-py",
    project_urls={
        "Bug Tracker": "https://github.com/saezlab/progeny-py/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "progeny"},
    packages=setuptools.find_packages(where="progeny"),
    python_requires=">=3.6",
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
