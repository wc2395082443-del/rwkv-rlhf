#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, json, argparse, time, re
from pathlib import Path
from typing import Optional

import torch
from tqdm import tqdm
from math_verify import parse as math_parse, verify as math_verify

CODE_DIR = '/root/RWKV-LM/RWKV7-statetuning_math500_hb_eval64_v1'
sys.path.insert(0, CODE_DIR)
from main import load_train_model_rwkv7_cuda, load_time_state_only, load_infer_model_albatross
from reference.utils import TRIE_TOKENIZER
from infer import AlbatrossBatchInference
from train import GRPOConfig

DEFAULT_MODEL = '/dev/shm/rwkv_models/rwkv7-g1e-1.5b-20260309-ctx8192.pth'
DEFAULT_TOKENIZER = '/root/RWKV-LM/rwkv_vocab_v20230424.txt'
DEFAULT_MATH500 = '/root/autodl-tmp/data/math500/test.jsonl'
DEFAULT_TUNED = '/dev/shm/rwkv_runs/math500_hb_50_mb2_len1024_20260408_202202/ckpt_step50.pth'

def safe_decode(tok, ids):
    try:
        return tok.decode(ids, utf8_errors='replace')
    except TypeError:
        try:
            return tok.decode(ids)
        except Exception:
            return ''
    except Exception:
        return ''



def extract_number(text: str) -> Optional[str]:
    text = str(text)
    boxed = re.findall(r'\\boxed\{([^{}]+)\}', text)
    if boxed:
        text = boxed[-1]
    nums = re.findall(r'-?\d+(?:,\d{3})*(?:\.\d+)?', text)
    if not nums:
        return None
    return nums[-1].replace(',', '')


def numeric_equal(pred: str, gt: str) -> bool:
    a, b = extract_number(pred), extract_number(gt)
    if a is None or b is None:
        return str(pred).strip().lower() == str(gt).strip().lower()
    try:
        return abs(float(a) - float(b)) < 1e-6
    except Exception:
        return a == b


def math_verify_equal(completion: str, gt: str) -> bool:
    """Match Albatross MATH500 judging: parse full completion against boxed gold."""
    try:
        gold = math_parse(f"$\\boxed{{{str(gt).strip()}}}$")
        pred = math_parse(str(completion))
        return bool(pred and math_verify(gold, pred, strict=False))
    except Exception:
        return False


def read_rows(path: str):
    text = Path(path).read_text(encoding='utf-8').strip()
    if not text:
        return []
    if text[0] == '[':
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError(f'expected list json in {path}')
        return data
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def make_cfg(args):
    return GRPOConfig(
        num_questions=1, samples_per_question=args.sample_n, max_new_tokens=args.max_new_tokens,
        temperature=args.sample_temperature, top_p=args.sample_top_p, top_k=args.sample_top_k,
        eval_temperature=args.greedy_temperature, eval_top_p=args.greedy_top_p, eval_top_k=args.greedy_top_k,
        ppo_epochs=1, micro_batch=1, lr=1e-4, grad_clip=1.0,
        min_tokens=0, max_tokens=args.max_new_tokens, length_weight=0.0,
        zstd_threshold=2.5, zstd_penalty_weight=0.0, ngram_penalty=0.0,
        answer_judge='auto', neg_adv_weight=1.0, kl_coef=0.0, kl_mode='none',
        time_state_l2=0.0, time_state_clamp=0.0, log_interval=1, save_interval=999999,
        eval_interval=999999, eval_sample_ratio=1.0,
        hard_buffer_ttl=0, hard_buffer_cooldown=0, hard_buffer_target_samples=0,
        hard_buffer_group_size=1, hard_buffer_extra_lr_scale=0.0, hard_buffer_adv_clip=0.0)


def load_engine(args):
    os.environ['RWKV_HEAD_SIZE_A']='64'
    os.environ['RWKV_MY_TESTING']='x070'
    os.environ['RWKV_TRAIN_TYPE']='state'
    os.environ['RWKV_CTXLEN']=str(args.ctx_len)
    os.environ['FUSED_KERNEL']='0'
    os.environ['WKV']='cuda'
    tok = TRIE_TOKENIZER(args.tokenizer)
    train_model, _ = load_train_model_rwkv7_cuda(args.model, 'cuda', args.ctx_len)
    state_loaded = False
    if args.state_init:
        state_loaded = load_time_state_only(train_model, args.state_init)
    train_model.eval()
    base_name = args.model[:-4] if args.model.endswith('.pth') else args.model
    infer_model, _ = load_infer_model_albatross(base_name)
    cfg = make_cfg(args)
    engine = AlbatrossBatchInference(infer_model, train_model, tok.encode, lambda ids: safe_decode(tok, ids), 'cuda', cfg)
    return tok, engine, state_loaded


def prompt_for(q: str, style: str = 'fake_think') -> str:
    problem = str(q).strip().replace('\r\n', '\n')
    if style == 'fake_think':
        return f'User: {problem}\n\nAssistant: <think></think'
    if style == 'plain':
        return f'User: {problem}\n\nAssistant:'
    raise ValueError(f'unknown prompt_style={style}')


def gen_group(engine, tok, prompt: str, n: int, max_new: int, temperature: float, top_p: float, top_k: int):
    comp, _, texts, _ = engine.generate_group_parallel(
        [tok.encode(prompt)], group_size=n, max_new_tokens=max_new,
        temperature=temperature, top_p=top_p, top_k=top_k,
        stop_on_user=True, stop_on_boxed=False, stop_on_repeat_ngram=False,
        post_trunc_append='', post_trunc_max_tokens=0)
    if texts:
        return texts
    return [tok.decode(c) for c in comp]


def eval_math(args):
    rows = read_rows(args.data)
    if args.limit:
        rows = rows[:args.limit]
    tok, engine, state_loaded = load_engine(args)
    total = 0
    greedy_ok = 0
    correct_generations = 0
    pass_ok = 0
    greedy_lens = []
    sample_lens = []
    generation_rows = []
    examples = []
    t0 = time.time()
    for r in tqdm(rows, desc=f'areal_math500_n{args.sample_n}'):
        q = r.get('problem') or r.get('question') or r.get('input')
        gt = r.get('ground_truth') or r.get('answer') or r.get('target') or r.get('solution') or r.get('original_answer')
        if q is None or gt is None:
            continue
        prompt = prompt_for(q, args.prompt_style)
        greedy_text = None
        g_ok = None
        if not args.skip_greedy:
            greedy_text = gen_group(engine, tok, prompt, 1, args.max_new_tokens, args.greedy_temperature, args.greedy_top_p, args.greedy_top_k)[0]
            g_ok = math_verify_equal(greedy_text, gt)
        sample_texts = gen_group(engine, tok, prompt, args.sample_n, args.max_new_tokens, args.sample_temperature, args.sample_top_p, args.sample_top_k)
        sample_flags = [math_verify_equal(x, gt) for x in sample_texts]
        if g_ok is not None:
            greedy_ok += int(g_ok)
        correct_generations += sum(int(x) for x in sample_flags)
        pass_ok += int(any(sample_flags))
        total += 1
        if greedy_text is not None:
            greedy_lens.append(len(tok.encode(greedy_text)))
        sample_token_lens = [len(tok.encode(x)) for x in sample_texts]
        sample_lens.extend(sample_token_lens)
        for sample_id, (text, ok, tok_len) in enumerate(zip(sample_texts, sample_flags, sample_token_lens)):
            generation_rows.append({
                'task_index': total - 1,
                'sample_id': sample_id,
                'problem': q,
                'answer': gt,
                'completion': text,
                'correct': bool(ok),
                'generated_tokens': tok_len,
                'truncated': tok_len >= args.max_new_tokens,
            })
        if len(examples) < args.examples:
            examples.append({
                'gt': gt,
                'greedy_correct': g_ok,
                'greedy_pred_num': extract_number(greedy_text) if greedy_text is not None else None,
                'sample_correct_count': sum(int(x) for x in sample_flags),
                'sample_pred_nums': [extract_number(x) for x in sample_texts[:min(4, len(sample_texts))]],
            })
    generations_jsonl = str(Path(args.out).with_suffix('.generations.jsonl'))
    if args.write_generations:
        with open(generations_jsonl, 'w', encoding='utf-8') as f:
            for row in generation_rows:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
    result = {
        'benchmark': args.benchmark,
        'data': args.data,
        'count': total,
        'state_init': args.state_init,
        'state_loaded': state_loaded,
        'rollout': args.sample_n,
        'total_generations': total * args.sample_n,
        'correct_generations': correct_generations,
        'rollout_accuracy': correct_generations / max(total * args.sample_n, 1),
        'pass_at_rollout_accuracy': pass_ok / total if total else None,
        'greedy_acc': (greedy_ok / total if total and not args.skip_greedy else None),
        f'sample_pass@1_n{args.sample_n}': correct_generations / max(total * args.sample_n, 1),
        f'pass@{args.sample_n}': pass_ok / total if total else None,
        'avg_greedy_len': sum(greedy_lens) / len(greedy_lens) if greedy_lens else None,
        'avg_sample_len': sum(sample_lens) / len(sample_lens) if sample_lens else None,
        'max_new_tokens': args.max_new_tokens,
        'greedy_temperature': args.greedy_temperature,
        'greedy_top_p': args.greedy_top_p,
        'greedy_top_k': args.greedy_top_k,
        'sample_temperature': args.sample_temperature,
        'sample_top_p': args.sample_top_p,
        'sample_top_k': args.sample_top_k,
        'sample_n': args.sample_n,
        'prompt_style': args.prompt_style,
        'verifier': 'math_verify.parse + verify(strict=False), boxed gold',
        'sampler_order': 'target Albatross temperature=1.0 top_k=32 top_p=0.28',
        'stop': 'eod/user_stop/max_tokens; stop_on_boxed=False; repeat_ngram_stop=False',
        'generations_jsonl': generations_jsonl if args.write_generations else None,
        'elapsed_sec': time.time() - t0,
        'examples': examples,
        'note': 'Albatross-aligned MATH500 eval: fake_think prompt, math_verify judge, rollout_accuracy and pass_at_rollout_accuracy.'
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        f.write(json.dumps(result, ensure_ascii=False) + '\n')
    print(json.dumps(result, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default=DEFAULT_MODEL)
    ap.add_argument('--state_init', default='')
    ap.add_argument('--tokenizer', default=DEFAULT_TOKENIZER)
    ap.add_argument('--data', default=DEFAULT_MATH500)
    ap.add_argument('--benchmark', default='math500_albatross_aligned')
    ap.add_argument('--out', default='/dev/shm/rwkv_runs/official_like_eval/math500_albatross_aligned.jsonl')
    ap.add_argument('--ctx_len', type=int, default=8192)
    ap.add_argument('--max_new_tokens', type=int, default=1500)
    ap.add_argument('--sample_n', type=int, default=4)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--examples', type=int, default=5)
    ap.add_argument('--prompt_style', choices=('fake_think', 'plain'), default='fake_think')
    ap.add_argument('--skip_greedy', action='store_true', default=True)
    ap.add_argument('--run_greedy', action='store_false', dest='skip_greedy')
    ap.add_argument('--write_generations', action='store_true', default=True)
    ap.add_argument('--no_write_generations', action='store_false', dest='write_generations')
    ap.add_argument('--greedy_temperature', type=float, default=1.0)
    ap.add_argument('--greedy_top_p', type=float, default=0.28)
    ap.add_argument('--greedy_top_k', type=int, default=32)
    ap.add_argument('--sample_temperature', type=float, default=1.0)
    ap.add_argument('--sample_top_p', type=float, default=0.28)
    ap.add_argument('--sample_top_k', type=int, default=32)
    args = ap.parse_args()
    eval_math(args)

if __name__ == '__main__':
    main()
