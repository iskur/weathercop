set CC=icl
set LD=xilink

python setup.py build --force --compiler=intelemw install --old-and-unmanageable
rem python setup.py install
if errorlevel 1 exit 1
