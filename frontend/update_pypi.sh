# Nothing fancy, run this script in the folder where it's located.

# Put your PyPi token in $HOME/.pypirc

PYTHON=`which python3.10`

echo "Upgrade build and twine"
$PYTHON -m pip install --upgrade build
$PYTHON -m pip install --upgrade twine

echo "Build package"
cd heco
$PYTHON -m build

echo "Upload package to PyPi"
$PYTHON -m twine upload dist/*

echo "Done."