#!/usr/bin/env python3
import os, sys, json, time, argparse, types
from pathlib import Path

import torch

# Keep env aligned with training before importing the training module.
os.environ.setdefault('RWKV_MY_TESTING', 'x070')
os.environ.setdefault('RWKV_CTXLEN', '8192')
os.environ.setdefault('RWKV_HEAD_SIZE', '64')
os.environ.setdefault('RWKV_FLOAT_MODE', 'bf16')
os.environ.setdefault('RWKV_JIT_ON', '0')

import train_rl_thinking_cot_strict_boxedprompt_schedopt_earlystop as T


def read_jsonl(path, limit=0):
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if limit and len(rows) >= limit:
                    break
    return rows


def get_answer(row):
    for k in ('answer', 'target', 'final_answer', 'ground_truth'):
        if k in row and row[k] is not None:
            return str(row[k])
    return ''


def make_args(load_model, tokenizer, ctx_len):
    return types.SimpleNamespace(
        load_model=load_model,
        tokenizer=tokenizer,
        ctx_len=ctx_len,
        head_size=64,
        my_testing='x070',
        precision='bf16',
        weight_decay=0.0,
        beta1=0.9,
        beta2=0.99,
        adam_eps=1e-8,
        fp32_master_optimizer=0,
        lr_init=0.0,
        lr_final=0.0,
        betas=(0.9, 0.99),
        train_stage=0,
        grad_cp=0,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--tokenizer', required=True)
    ap.add_argument('--eval_jsonl', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--max_new_tokens', type=int, default=2048)
    ap.add_argument('--temperature', type=float, default=0.3)
    ap.add_argument('--top_p', type=float, default=0.4)
    ap.add_argument('--top_k', type=int, default=500)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--seed', type=int, default=424242)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    sd = T._normalize_state_dict(T._torch_load_weights(args.model))
    n_layer, n_embd, vocab_size, dim_ffn = T._infer_arch(sd)
    margs = make_args(args.model, args.tokenizer, max(8192, args.max_new_tokens + 4096))
    margs.n_layer, margs.n_embd, margs.vocab_size, margs.dim_ffn = n_layer, n_embd, vocab_size, dim_ffn
    margs.dim_att = n_embd

    data = read_jsonl(args.eval_jsonl, args.limit)
    cfg = T.GRPOConfig(max_new_tokens=args.max_new_tokens, eval_temperature=args.temperature, eval_top_p=args.top_p, eval_top_k=args.top_k)
    cfg.max_tokens = args.max_new_tokens
    cfg.min_tokens = 200
    cfg.rollout_forward_batch = args.batch

    model = T.RWKVGRPOModel(args=margs, rl_cfg=cfg, train_data=data[:1], test_data=data, full_test_data=data)
    model = model.to('cuda', dtype=T._rwkv_float_dtype()).eval()
    encode, decode = model._build_tokenizer()
    model.prepare_stateful_rollout()
    infer = T.TrainTempBatchInference(model, model, encode, decode, 'cuda', cfg)

    total = loose = strict = fmt = eod = trunc = repeat = multilingual = degen = 0
    toks = 0
    examples = []
    responses_path = Path(args.out_dir, 'responses.jsonl')
    response_f = responses_path.open('w', encoding='utf-8')
    t0 = time.time()
    with response_f, torch.no_grad():
        for start in range(0, len(data), args.batch):
            rows = data[start:start+args.batch]
            prompts = []
            for row in rows:
                ids = [0] + encode(T.build_thinking_prompt(row.get('problem','')))
                max_prompt_len = int(margs.ctx_len) - args.max_new_tokens - 4
                if len(ids) > max_prompt_len:
                    ids = ids[-max(64, max_prompt_len):]
                prompts.append(ids)
            comp_tokens, _, comp_texts, truncated, ended_eod = infer.generate_group_parallel(
                prompt_tokens_list=prompts,
                group_size=1,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                stop_on_token_zero=True,
                stop_on_user=False,
                stop_on_repeat_ngram=True,
                post_trunc_append='',
                post_trunc_max_tokens=0,
            )
            for i, row in enumerate(rows):
                reward, is_correct, is_format_correct, details = T.calculate_boxed_think_reward_details(
                    text=comp_texts[i],
                    ground_truth=get_answer(row),
                    token_length=len(comp_tokens[i]),
                    min_tokens=200,
                    max_tokens=args.max_new_tokens,
                    length_weight=0.0,
                    repeat_ngram=False,
                    repeat_penalty=0.0,
                    zstd_penalty_weight=0.0,
                    ended_eod=ended_eod[i],
                    truncated=truncated[i],
                    overlong_buffer_tokens=0,
                    overlong_penalty_factor=0.0,
                )
                q = details.get('quality_gate') or {}
                total += 1
                loose += int(is_correct)
                fmt += int(is_format_correct)
                strict += int(is_correct and is_format_correct)
                eod += int(ended_eod[i])
                trunc += int(truncated[i])
                repeat += int(q.get('repetitive', False))
                multilingual += int(q.get('multilingual', False))
                degen += int(not q.get('ok', True))
                toks += len(comp_tokens[i]) - int(ended_eod[i] and comp_tokens[i] and comp_tokens[i][-1] == 0)
                record = {
                    'index': total - 1,
                    'problem': row.get('problem',''),
                    'ground_truth': get_answer(row),
                    'response': comp_texts[i],
                    'is_correct': bool(is_correct),
                    'strict_format': bool(is_format_correct),
                    'strict_correct': bool(is_correct and is_format_correct),
                    'ended_eod': bool(ended_eod[i]),
                    'truncated': bool(truncated[i]),
                    'token_length': int(len(comp_tokens[i])),
                    'quality_gate': q,
                    'reward_details': details,
                }
                response_f.write(json.dumps(record, ensure_ascii=False) + '\n')
                if len(examples) < 5:
                    ex = dict(record)
                    ex['problem'] = ex['problem'][:300]
                    ex['text'] = ex.pop('response')[:1500]
                    ex['format'] = ex.pop('strict_format')
                    examples.append(ex)
            print(f'progress {total}/{len(data)} loose={loose/max(1,total):.4f} strict={strict/max(1,total):.4f} trunc={trunc/max(1,total):.4f} len={toks/max(1,total):.1f}', flush=True)

    summary = {
        'model': args.model,
        'eval_jsonl': args.eval_jsonl,
        'num_tasks': total,
        'loose_accuracy': loose/max(1,total),
        'strict_accuracy': strict/max(1,total),
        'strict_format_rate': fmt/max(1,total),
        'eod_rate': eod/max(1,total),
        'truncated_rate': trunc/max(1,total),
        'repeat_rate': repeat/max(1,total),
        'multilingual_rate': multilingual/max(1,total),
        'degeneration_rate': degen/max(1,total),
        'mean_generated_tokens': toks/max(1,total),
        'max_new_tokens': args.max_new_tokens,
        'temperature': args.temperature,
        'top_p': args.top_p,
        'top_k': args.top_k,
        'elapsed_sec': time.time() - t0,
        'responses_path': str(responses_path),
    }
    Path(args.out_dir, 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    Path(args.out_dir, 'examples.json').write_text(json.dumps(examples, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)

if __name__ == '__main__':
    main()
