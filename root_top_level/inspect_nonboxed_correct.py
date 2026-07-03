import re
import pyarrow.parquet as pq
from datasets import load_from_disk
from math_verify import parse, verify

COMP_P = '/root/autodl-tmp/trl-grpo-repro/outputs/qwen2_deepmath_grpo_9q_8roll_lr1e6_ga8_log/completions/completions_00001.parquet'
DS_P = '/root/autodl-tmp/official_repro_assets/DeepMath-103K-trl-deepmath_zero'

def prompt_to_key(prompt_obj):
    if isinstance(prompt_obj, list):
        parts = []
        for x in prompt_obj:
            if isinstance(x, dict):
                parts.append(f"{x.get('role','')}::{x.get('content','')}")
            else:
                parts.append(str(x))
        return '||'.join(parts)
    return str(prompt_obj)

# load first 9 train rows used in the run
train = load_from_disk(DS_P)['train'].select(range(9))
gt_map = {}
for row in train:
    gt_map[prompt_to_key(row['prompt'])] = str(row['ground_truth'])

# load completions parquet
t = pq.read_table(COMP_P)
d = {name: t[name].to_pylist() for name in t.column_names}
rows = len(d['completion'])

nonboxed_total = 0
nonboxed_mathverify_correct = 0
nonboxed_regex_lastnum_correct = 0
examples = []

num_pat = re.compile(r'(-?\d+(?:/\d+)?(?:\.\d+)?)')

for prompt_obj, completion in zip(d['prompt'], d['completion']):
    text = str(completion)
    if '\\boxed' in text:
        continue
    nonboxed_total += 1
    gt = gt_map.get(prompt_to_key(prompt_obj))
    if gt is None:
        continue

    mv_correct = False
    try:
        pred = parse(text)
        gold = parse(f'\\boxed{{$' + gt + '$}}')
        mv_correct = bool(verify(gold, pred, timeout_seconds=3))
    except Exception:
        mv_correct = False
    if mv_correct:
        nonboxed_mathverify_correct += 1
        if len(examples) < 5:
            examples.append((gt, text[:500]))

    # cheap proxy: compare last numeric-ish token to gold via math_verify too
    nums = num_pat.findall(text)
    if nums:
        candidate = nums[-1]
        try:
            pred2 = parse(candidate)
            gold2 = parse(f'\\boxed{{$' + gt + '$}}')
            if bool(verify(gold2, pred2, timeout_seconds=3)):
                nonboxed_regex_lastnum_correct += 1
        except Exception:
            pass

print('rows=', rows)
print('nonboxed_total=', nonboxed_total)
print('nonboxed_mathverify_correct=', nonboxed_mathverify_correct)
print('nonboxed_lastnum_correct=', nonboxed_regex_lastnum_correct)
print('examples=')
for gt, txt in examples:
    print('--- gt=', gt)
    print(txt.replace('\n', '\\n'))
