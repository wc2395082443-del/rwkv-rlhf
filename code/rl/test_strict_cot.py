#!/usr/bin/env python3
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from infer import AlbatrossBatchInference
from reward import calculate_reward_details, math_equal


cases = [
    ('thinking</think><answer>\\boxed{78}</answer>', '78', True, '78', True, True, 1.0),
    ('</think><answer>\\boxed{78}</answer>', '78', True, '78', True, False, 0.0),
    ('   </think><answer>78</answer>', '78', True, '78', True, False, 0.0),
    ('thinking</think><answer>78</answer>', '78', True, '78', True, True, 1.0),
    ('thinking</think><answer>\\boxed{222}</answer>', '78', True, '222', False, True, 0.0),
    ('thinking</think><answer>\\boxed{78}</answer>', '78', False, '78', True, False, 0.0),
    ('thinking</think><answer>\\boxed{78}</answer> junk', '78', True, '78', True, False, 0.0),
    ('thinking</think>\\boxed{78}', '78', True, '78', True, False, 0.0),
    ('thinking</think><answer>\\boxed{78}</answer><answer>78</answer>', '78', True, '78', True, False, 0.0),
]
for response, gold, ended_eod, extracted, correct, fmt, expected_reward in cases:
    reward, got_correct, got_fmt, details = calculate_reward_details(
        response,
        gold,
        32,
        min_tokens=1,
        max_tokens=64,
        ended_eod=ended_eod,
    )
    assert details['extracted_answer'] == extracted, (response, details)
    assert got_correct is correct, (response, details)
    assert got_fmt is fmt, (response, details)
    assert reward == expected_reward, (response, details)
    assert details['correct_reward'] == 0.0, (response, details)
    assert details['format_reward'] == float(correct and fmt), (response, details)
assert not math_equal('2020!', '2020! + 2020')


class FakeInternalInfer(AlbatrossBatchInference):
    def __init__(self, first_token):
        self.cfg = SimpleNamespace(tune_mode='full')
        self.first_token = first_token
        self.infer_model = SimpleNamespace()
        self._stop_token_id_seqs = []
        self.encode = lambda text: [7]
        self.decode = lambda ids: ''.join({5: 'M', 6: 'A', 7: '\n', 8: 'X'}.get(x, '') for x in ids if x > 0)

    def prime_prompts(self, prompt_tokens_list):
        logits = torch.full((len(prompt_tokens_list), 16), -100.0)
        logits[:, self.first_token] = 100.0
        return logits, None

    def _forward_last_logits_full(self, full_sequences):
        logits = torch.full((len(full_sequences), 16), -100.0)
        logits[:, 0] = 100.0
        return logits


engine = FakeInternalInfer(first_token=0)
tokens, logps, texts, truncated, ended_eod = engine.generate_group_parallel(
    [[1]], group_size=1, max_new_tokens=4, temperature=0.0, top_p=1.0, top_k=0,
)
assert tokens == [[0]] and len(logps[0]) == 1 and texts == ['']
assert truncated == [False] and ended_eod == [True]

engine = FakeInternalInfer(first_token=5)
tokens, _, texts, truncated, ended_eod = engine.generate_group_parallel(
    [[1]], group_size=1, max_new_tokens=1, temperature=0.0, top_p=1.0, top_k=0,
)
assert tokens == [[5]] and texts == ['M'] and truncated == [True] and ended_eod == [False]


spec = importlib.util.spec_from_file_location('format_b_eval', ROOT / 'eval_thinking_cot_strict.py')
format_b_eval = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = format_b_eval
spec.loader.exec_module(format_b_eval)

valid = {
    'completion': 'thinking</think><answer>\\boxed{1}</answer>',
    'answer': '1',
    'ended_eod': True,
}
verified = format_b_eval.verify_one(valid)
assert verified['correct'] and verified['strict_correct'] and verified['strict_format'], verified
assert verified['has_nonempty_thought'], verified

empty_thought = dict(valid, completion='</think><answer>\\boxed{1}</answer>')
verified = format_b_eval.verify_one(empty_thought)
assert verified['correct'] and not verified['strict_correct'] and not verified['strict_format'], verified
assert not verified['has_nonempty_thought'], verified

no_eod = dict(valid, ended_eod=False)
verified = format_b_eval.verify_one(no_eod)
assert verified['correct'] and not verified['strict_correct'] and not verified['strict_format'], verified

trailing = dict(valid, completion=valid['completion'] + ' junk')
verified = format_b_eval.verify_one(trailing)
assert verified['correct'] and not verified['strict_format'], verified

wrong = dict(valid, completion='thinking mentions 1</think><answer>\\boxed{2}</answer>')
verified = format_b_eval.verify_one(wrong)
assert not verified['correct'] and verified['strict_format'], verified

print('ALL_FORMAT_B_EOS_TESTS_OK')
