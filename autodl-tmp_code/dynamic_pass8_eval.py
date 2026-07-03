import os
import sys
import json
import time
import types
import argparse
import importlib.util
from pathlib import Path

import torch

BASE_DIR = Path('/root/RWKV-LM/RWKV-v7/train_temp')
BASELINE_DIR = Path('/root/RWKV-LM/RWKV7-statetuning_hardbuffer_v1')
os.chdir(BASE_DIR)
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
if str(BASELINE_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINE_DIR))

import train_rl_baseline as mod

reward_spec = importlib.util.spec_from_file_location('baseline_reward', BASELINE_DIR / 'reward.py')
reward_mod = importlib.util.module_from_spec(reward_spec)
reward_spec.loader.exec_module(reward_mod)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--eval_jsonl', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--tokenizer', default='/root/RWKV-LM/RWKV-v7/rwkv_vocab_v20230424.txt')
    ap.add_argument('--train_jsonl', default='/root/RWKV-LM/RWKV7-statetuning/gsm8k_train_formatted.jsonl')
    ap.add_argument('--strategy', default='deepspeed_stage_3_offload')
    ap.add_argument('--precision', default='bf16')
    ap.add_argument('--use_stateful_rollout', type=int, default=1)
    ap.add_argument('--max_new_tokens', type=int, default=768)
    ap.add_argument('--micro_batch', type=int, default=8)
    ap.add_argument('--rollout_forward_batch', type=int, default=192)
    ap.add_argument('--temperature', type=float, default=1.0)
    ap.add_argument('--top_p', type=float, default=0.6)
    ap.add_argument('--top_k', type=int, default=0)
    ap.add_argument('--group_size', type=int, default=8)
    ap.add_argument('--chunk_size', type=int, default=24)
    ap.add_argument('--random_seed', type=int, default=42)
    args0 = ap.parse_args()

    out_dir = Path(args0.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sys.argv = [
        'pass8_eval',
        '--load_model', args0.model,
        '--proj_dir', str(out_dir),
        '--tokenizer', args0.tokenizer,
        '--train_jsonl', args0.train_jsonl,
        '--eval_jsonl', args0.eval_jsonl,
        '--strategy', args0.strategy,
        '--precision', args0.precision,
        '--use_stateful_rollout', str(int(args0.use_stateful_rollout)),
        '--max_new_tokens', str(int(args0.max_new_tokens)),
        '--micro_batch', str(int(args0.micro_batch)),
        '--rollout_forward_batch', str(int(args0.rollout_forward_batch)),
        '--random_seed', str(int(args0.random_seed)),
    ]
    args = mod.parse_args()
    mod.set_seed(int(args.random_seed))

    rwkv_precision = {'32': 'fp32', 32: 'fp32', '16': 'fp16', 16: 'fp16'}.get(args.precision, args.precision)
    os.environ['RWKV_MY_TESTING'] = args.my_testing
    os.environ['RWKV_CTXLEN'] = str(int(args.ctx_len))
    os.environ['RWKV_HEAD_SIZE'] = str(int(args.head_size))
    os.environ['RWKV_FLOAT_MODE'] = rwkv_precision
    os.environ['RWKV_JIT_ON'] = '0' if 'deepspeed_stage_3' in str(args.strategy) else '1'

    def read_jsonl(path):
        rows = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    sd = mod._normalize_state_dict(mod._torch_load_weights(args.load_model))
    args.n_layer, args.n_embd, args.vocab_size, args.dim_ffn = mod._infer_arch(sd)
    args.dim_att = args.n_embd

    tok = mod.TRIE_TOKENIZER(args.tokenizer)
    def encode_fn(s):
        return tok.encode(s)
    def decode_fn(ids):
        try:
            return tok.decode(ids, utf8_errors='replace')
        except Exception:
            try:
                return tok.decode(ids)
            except Exception:
                try:
                    return tok.decodeBytes(ids).decode('utf-8', errors='replace')
                except Exception:
                    return ''.join(chr(int(x) % 256) for x in ids)

    train_model = mod.PaddedRWKV(args)
    train_model.load_state_dict(sd, strict=True)
    train_model = mod._cast_ref_model_dtype(train_model.to('cuda'))
    train_model.eval()
    for p in train_model.parameters():
        p.requires_grad = False

    rollout_args = types.SimpleNamespace(
        MODEL_NAME=str(Path(args.load_model).with_suffix('')),
        vocab_size=int(args.vocab_size),
    )
    rollout_model = mod.RWKV_x070(rollout_args)
    rollout_model.eval()

    cfg = types.SimpleNamespace(
        tune_mode='state',
        rollout_forward_batch=int(args.rollout_forward_batch),
    )
    infer = mod.TrainTempBatchInference(
        infer_model=rollout_model,
        train_model=train_model,
        encode_fn=encode_fn,
        decode_fn=decode_fn,
        device='cuda',
        cfg=cfg,
    )

    data = read_jsonl(args0.eval_jsonl)
    summary = {
        'model': args0.model,
        'eval_jsonl': args0.eval_jsonl,
        'group_size': int(args0.group_size),
        'temperature': float(args0.temperature),
        'top_p': float(args0.top_p),
        'top_k': int(args0.top_k),
        'total_questions': len(data),
    }

    sample_correct = 0
    sample_total = 0
    pass8_correct = 0
    first_correct = 0
    num_correct_hist = {k: 0 for k in range(9)}
    out_jsonl = out_dir / 'pass8_eval.jsonl'
    if out_jsonl.exists():
        out_jsonl.unlink()

    t0 = time.time()
    with open(out_jsonl, 'w', encoding='utf-8') as fout:
        for start in range(0, len(data), int(args0.chunk_size)):
            ex_list = data[start:start + int(args0.chunk_size)]
            problems = [ex.get('problem', '') for ex in ex_list]
            answers = [ex.get('solution', ex.get('ground_truth', ex.get('answer', ex.get('original_answer', '')))) for ex in ex_list]
            prompt_strs = [mod._baseline_mod.build_prompt(p) for p in problems]
            prompt_tokens_list = []
            for ps in prompt_strs:
                ids = encode_fn(ps)
                max_prompt_len = int(args.ctx_len) - int(args0.max_new_tokens) - 4
                max_prompt_len = max(64, max_prompt_len)
                if len(ids) > max_prompt_len:
                    ids = ids[-max_prompt_len:]
                prompt_tokens_list.append(ids)

            comp_tokens_list, _, comp_texts_list, truncated_list = infer.generate_group_parallel(
                prompt_tokens_list=prompt_tokens_list,
                group_size=int(args0.group_size),
                max_new_tokens=int(args0.max_new_tokens),
                temperature=float(args0.temperature),
                top_p=float(args0.top_p),
                top_k=int(args0.top_k),
            )

            for i, (problem, answer) in enumerate(zip(problems, answers)):
                start_idx = i * int(args0.group_size)
                end_idx = start_idx + int(args0.group_size)
                sample_records = []
                any_correct = False
                for j in range(start_idx, end_idx):
                    comp_text = comp_texts_list[j]
                    comp_tokens = comp_tokens_list[j]
                    truncated = bool(truncated_list[j])
                    reward, is_correct, is_format_correct, reward_details = reward_mod.calculate_reward_details(
                        text=comp_text,
                        ground_truth=answer,
                        token_length=len(comp_tokens),
                        min_tokens=200,
                        max_tokens=int(args0.max_new_tokens),
                        length_weight=0.0,
                        repeat_ngram=False,
                        repeat_penalty=0.0,
                        zstd_threshold=2.5,
                        zstd_penalty_weight=0.0,
                    )
                    is_correct = bool(is_correct)
                    any_correct = any_correct or is_correct
                    sample_correct += int(is_correct)
                    sample_total += 1
                    sample_records.append({
                        'sample_idx': j - start_idx,
                        'is_correct': is_correct,
                        'is_format_correct': bool(is_format_correct),
                        'truncated': truncated,
                        'pred_extracted': reward_details.get('extracted_answer'),
                        'gt_extracted': reward_details.get('ground_truth_answer'),
                        'gen_len': len(comp_tokens),
                        'response': comp_text,
                    })
                pass8_correct += int(any_correct)
                first_correct += int(bool(sample_records and sample_records[0]['is_correct']))
                num_correct = sum(int(x['is_correct']) for x in sample_records)
                num_correct_hist[int(num_correct)] += 1
                rec = {
                    'problem': problem,
                    'ground_truth': answer,
                    'pass8_correct': bool(any_correct),
                    'num_correct': num_correct,
                    'samples': sample_records,
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + '\n')
            torch.cuda.empty_cache()

    summary['sample_avg_acc'] = sample_correct / sample_total if sample_total else 0.0
    summary['pass8'] = pass8_correct / len(data) if data else 0.0
    summary['first_sample_acc'] = first_correct / len(data) if data else 0.0
    summary['questions_with_any_correct'] = pass8_correct
    summary['questions_all_wrong'] = len(data) - pass8_correct
    summary['num_correct_hist'] = num_correct_hist
    summary['elapsed_sec'] = time.time() - t0
    with open(out_dir / 'summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False))

if __name__ == '__main__':
    main()
