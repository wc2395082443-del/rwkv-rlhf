import re
import pyarrow.parquet as pq
p = '/root/autodl-tmp/trl-grpo-repro/outputs/qwen2_deepmath_grpo_9q_8roll_lr1e6_ga8_log/completions/completions_00001.parquet'
t = pq.read_table(p)
d = {name: t[name].to_pylist() for name in t.column_names}
print('columns=', t.column_names)
rows = len(next(iter(d.values()))) if d else 0
print('rows=', rows)
completion = [str(x) for x in d['completion']]
boxed_count = [len(re.findall(r'\\boxed', x)) for x in completion]
print('boxed0=', sum(x == 0 for x in boxed_count))
print('boxed1=', sum(x == 1 for x in boxed_count))
print('boxed_gt1=', sum(x > 1 for x in boxed_count))
reward_cols = [c for c in t.column_names if 'reward' in c.lower()]
print('reward_cols=', reward_cols)
for c in reward_cols:
    vals = d[c]
    cnt = {}
    for v in vals:
        cnt[v] = cnt.get(v, 0) + 1
    print(c, cnt)
print('sample_nonboxed=')
shown = 0
for text, bc in zip(completion, boxed_count):
    if bc == 0:
        print('---')
        print(text[:700].replace('\n', '\\n'))
        shown += 1
        if shown >= 3:
            break
print('sample_boxed=')
shown = 0
for text, bc in zip(completion, boxed_count):
    if bc == 1:
        print('---')
        print(text[:700].replace('\n', '\\n'))
        shown += 1
        if shown >= 3:
            break
