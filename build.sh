CFLAGS="-I$PREFIX/include $CFLAGS" $PYTHON $RECIPE_DIR setup.py build --force --compiler=intelem install --old-and-unmanageable
# $PYTHON $RECIPE_DIR setup.py install     # Python command to install the script.
