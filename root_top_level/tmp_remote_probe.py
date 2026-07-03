import importlib.util as u
mods = ['trl','transformers','accelerate','datasets','torch','peft','bitsandbytes','math_verify']
print({m: bool(u.find_spec(m)) for m in mods})
