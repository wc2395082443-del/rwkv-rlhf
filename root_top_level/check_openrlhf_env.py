import importlib
mods = ['ray','torch','vllm','datasets','deepspeed','openrlhf']
for m in mods:
    importlib.import_module(m)
    print('OK', m)
