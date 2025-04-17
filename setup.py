from setuptools import setup, find_packages
import io
import os

here = os.path.abspath(os.path.dirname(__file__))

NAME = 'pdesolvers'

# Import version from file
version_file = open(os.path.join(here, 'VERSION'))
VERSION = version_file.read().strip()

DESCRIPTION = 'A package for solving partial differential equations'

# Import the README and use it as the long-description.
try:
    with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        LONG_DESCRIPTION = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

# Read requirements from requirements.txt
with open('requirements.txt') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name=NAME,
    version='0.1',
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',\
    author='Chelsea De Marseilla, Debdal Chowdhury',
    license='Apache License 2.0',
    packages=find_packages(),
    install_requires=requirements,
    extras_require={
        'package-tools': [
            'setuptools',
            'wheel',
            'twine'
        ]
    },
    url='https://github.com/GPUEngineering/PDESolvers',
)