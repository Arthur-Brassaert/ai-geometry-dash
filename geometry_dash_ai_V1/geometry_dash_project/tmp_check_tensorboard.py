import importlib, shutil, sys

try:
    importlib.import_module('tensorboard.program')
    print('tensorboard.program import OK')
except Exception as e:
    print('tensorboard.program import FAILED:', type(e).__name__, e)

print('tensorboard CLI on PATH:', shutil.which('tensorboard'))
print('sys.executable:', sys.executable)
