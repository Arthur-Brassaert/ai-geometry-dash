import importlib, time, os
from pathlib import Path

logdir = os.path.join(os.path.dirname(__file__), 'trained_models', 'logs')
print('Using logdir:', logdir)

tb = importlib.import_module('tensorboard.program')
server = tb.TensorBoard()
server.configure(argv=[None, '--logdir', logdir, '--port', '6006'])
url = server.launch()
print('Launched TensorBoard at', url)
print('Sleeping for 60s while server runs...')
try:
    time.sleep(60)
except KeyboardInterrupt:
    pass
print('Exiting')
