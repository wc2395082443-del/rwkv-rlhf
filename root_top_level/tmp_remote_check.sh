python3 - <<'PY'
import importlib
mods = ['trl','transformers','accelerate','datasets','torch','peft','bitsandbytes','math_verify']
for m in mods:
    try:
        mod = importlib.import_module(m)
        print(m, getattr(mod, '__version__', 'ok'))
    except Exception as e:
        print(m, 'MISSING', e)
PY
echo '=== models ==='
find /root/autodl-tmp/models -maxdepth 2 -type f \( -name 'config.json' -o -name 'model.safetensors.index.json' -o -name '*.pth' \) 2>/dev/null | head -n 80
