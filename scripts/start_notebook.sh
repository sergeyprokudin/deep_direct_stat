#!/usr/bin/env bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd)"
PYENV="${SCRIPT_DIR}/../py_env"
source ${PYENV}/bin/activate

python -m ipykernel install --user --name=py_env

jupyter notebook
