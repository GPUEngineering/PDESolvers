#!/bin/bash
# Make sure the tests will FAIL if it has to
set -euxo pipefail

export PYTHONPATH=$(pwd)

# Set up your Python environment
pip install virtualenv
virtualenv -p python3.13 venv
source venv/bin/activate

pip install -r requirements.txt

pytest