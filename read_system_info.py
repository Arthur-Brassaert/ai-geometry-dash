import zipfile
p = r'G:\test\ai-geometry-dash\geometry_dash_ai_v1\geometry_dash_project\trained_models\best_model.zip'
with zipfile.ZipFile(p,'r') as z:
    print('Files in zip:')
    print('\n'.join(z.namelist()))
    print('\n--- system_info.txt ---')
    try:
        print(z.read('system_info.txt').decode('utf-8'))
    except KeyError:
        print('system_info.txt not found inside zip')
