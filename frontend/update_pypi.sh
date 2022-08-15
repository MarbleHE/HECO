#!/bin/bash

# Put your PyPi token in $HOME/.pypirc

PACKAGE_ROOT=$(dirname $(readlink -f "${BASH_SOURCE:-$0}"))
PYTHON=`which python3.10`

cd "$PACKAGE_ROOT/heco"

echo -e "\nDelete old release locally"
rm -r -i dist/*

echo -e "\nInstall and upgrade build and twine\n"
$PYTHON -m pip install --upgrade build
$PYTHON -m pip install --upgrade twine

echo -e "\nBuild package\n"
TMPFILE=$(mktemp)
cp pyproject.toml $TMPFILE
trap "mv $TMPFILE pyproject.toml" EXIT # Always restore pytpoject.toml
sed -i -e "s/\"xdsl @ git+https:\/\/github.com\/xdslproject\/xdsl.git@frontend\"/# [DIRECT DEPENDENCY REMOVED]/g" pyproject.toml
$PYTHON -m build

echo -e "\nUpload package to PyPi\n"
$PYTHON -m twine upload dist/*

echo -e "\nDone."