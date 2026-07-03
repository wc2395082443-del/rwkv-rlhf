import inspect, importlib.util
spec=importlib.util.spec_from_file_location('b','/root/RWKV-LM/RWKV7-statetuning_hardbuffer_v1/train.py')
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
print(m.GRPOConfig)
print(inspect.signature(m.GRPOConfig))

