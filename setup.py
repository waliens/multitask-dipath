import os
from setuptools import setup

__version__ = "0.0.1-alpha"

long_description = ""
if os.path.isfile("README.md"):
    with open("README.md", "r") as fh:
        long_description = fh.read()

setup(
    name='mtdp',
    version=__version__,
    description='Implementation of multi-task trained networks, including models pre-trained on digital pathology data',
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=['mtdp', 'mtdp.models'],
    classifiers=[
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'License :: OSI Approved',
        'Programming Language :: Python',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7'
    ],
    install_requires=['torch', 'torchvision', 'numpy'],
    license='LICENSE'
)

