import os
from setuptools import setup
from mtdp import __version__

long_description = ""
if os.path.isfile("README.md"):
    with open("README.md", "r") as fh:
        long_description = fh.read()

setup(
    name='Multitask pre-trained deep neural networks for digital pathology',
    version=__version__,
    description='Implementation of multitask networks',
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=['mtdp', 'mtdp.models'],
    # TODO  url='https://github.com/Neubias-WG5',
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

