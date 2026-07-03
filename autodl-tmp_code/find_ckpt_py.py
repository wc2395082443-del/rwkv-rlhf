import os
root='/root/autodl-tmp/baseline_hardbuffer_kl005_bestcfg_20260422_194041'
for dirpath, dirnames, filenames in os.walk(root):
    for fn in filenames:
        if any(fn.endswith(ext) for ext in ['.ckpt','.pth','.pt','.bin']) or 'checkpoint' in fn.lower():
            print(os.path.join(dirpath, fn))

