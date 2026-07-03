import os
roots = ['/root/verl/verl', '/root/verl/examples']
patterns = ['return -1', 'score = -1', 'reward = -1', '-1.0']
for root in roots:
    for dp, _, fs in os.walk(root):
        for f in fs:
            path = os.path.join(dp, f)
            try:
                with open(path, 'r', encoding='utf-8') as fh:
                    for i, line in enumerate(fh, 1):
                        if any(p in line for p in patterns):
                            print(f'{path}:{i}:{line.rstrip()}')
            except Exception:
                pass
