from setuptools import setup, find_packages

with open('README.md') as f:
    LONG_DESCRIPTION = f.read()

setup(
    name='pdesolvers',
    version='0.1',
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',\
    author='Chelsea De Marseilla, Debdal Chowdhury',
    packages=find_packages(),
    install_requires=[
        'matplotlib==3.9.2',
        'numpy==2.1.3',
        'scipy==1.14.1',
        'pandas==2.2.3',
        'pytest==8.3.4'
    ],
)