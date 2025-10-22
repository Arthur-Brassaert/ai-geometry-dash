import importlib, os, threading
logdir = os.path.join(os.path.dirname(__file__), 'trained_models', 'logs')
print('Starting TB for', logdir)
server_mod = importlib.import_module('tensorboard.program')
server = server_mod.TensorBoard()
server.configure(argv=[None, '--logdir', logdir, '--port', '6006', '--bind_all'])
url = server.launch()
print('Launched TB at', url)
# Block forever
threading.Event().wait()
