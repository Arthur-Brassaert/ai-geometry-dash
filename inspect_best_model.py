import os
import datetime

p = r'G:\test\ai-geometry-dash\geometry_dash_ai_v1\geometry_dash_project\trained_models\best_model.zip'
log = r'G:\test\ai-geometry-dash\geometry_dash_ai_v1\geometry_dash_project\trained_models\last_improvements.log'

if not os.path.exists(p):
    print('best_model.zip not found at', p)
else:
    print('best_model.zip mtime ->', datetime.datetime.fromtimestamp(os.path.getmtime(p)).isoformat())

if not os.path.exists(log):
    print('last_improvements.log not found at', log)
else:
    print('\nlast_improvements.log (last 20 lines):')
    with open(log, 'r', encoding='utf-8') as f:
        lines = f.read().strip().splitlines()
    for ln in lines[-20:]:
        print(ln)
