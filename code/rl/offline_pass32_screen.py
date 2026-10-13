#!/usr/bin/env python3
import argparse, json, os, time, types, random
from pathlib import Path

import torch

import train_rl_thinking_cot_strict_boxedprompt_schedopt_earlystop as rl
from train_rl_thinking_cot_strict_boxedprompt_schedopt_earlystop import (
    RWKVGRPOModel,
    TrainTempBatchInference,
    GRPOConfig,
    build_thinking_prompt,
    calculate_boxed_think_reward_details,
)
from utils import read_jsonl


def get_answer(ex):
    for k in ("answer", "final_answer", "target", "gt", "solution"):
        v = ex.get(k)
        if v is not None and str(v).strip():
            return str(v)
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--load_model', required=True)
    ap.add_argument('--tokenizer', required=True)
    ap.add_argument('--input_jsonl', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--samples_per_question', type=int, default=32)
    ap.add_argument('--question_batch', type=int, default=8)
    ap.add_argument('--max_new_tokens', type=int, default=4096)
    ap.add_argument('--temperature', type=float, default=1.0)
    ap.add_argument('--top_p', type=float, default=1.0)
    ap.add_argument('--top_k', type=int, default=0)
    ap.add_argument('--rollout_forward_batch', type=int, default=512)
    ap.add_argument('--ctx_len', type=int, default=8192)
    ap.add_argument('--seed', type=int, default=45)
    args = ap.parse_args()

    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
    os.environ.setdefault('RWKV_MY_TESTING', 'x070')
    os.environ.setdefault('RWKV_CTXLEN', str(args.ctx_len))
    os.environ.setdefault('RWKV_HEAD_SIZE', '64')
    os.environ.setdefault('RWKV_FLOAT_MODE', 'bf16')
    os.environ.setdefault('RWKV_JIT_ON', '0')

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    mixed_path = out / 'train_pass32_mixed.jsonl'
    all_stats_path = out / 'pass32_screen_stats.jsonl'
    summary_path = out / 'pass32_screen_summary.json'

    data = read_jsonl(args.input_jsonl)
    end = len(data) if args.limit <= 0 else min(len(data), args.start + args.limit)
    data_slice = data[args.start:end]

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Infer architecture from checkpoint, matching the training script's RWKV args.
    sd = rl._torch_load_weights(args.load_model)
    arch_raw = rl._infer_arch(rl._normalize_state_dict(dict(sd)))
    if isinstance(arch_raw, dict):
        arch = arch_raw
    else:
        n_layer, n_embd, vocab_size, dim_ffn = arch_raw
        arch = {'n_embd': n_embd, 'n_layer': n_layer, 'vocab_size': vocab_size, 'dim_ffn': dim_ffn}
    model_args = types.SimpleNamespace(
        load_model=args.load_model,
        tokenizer=args.tokenizer,
        ctx_len=args.ctx_len,
        head_size=64,
        my_testing='x070',
        grad_cp=0,
        strategy='auto',
        weight_decay=0.0,
        lr_init=1e-7,
        betas=(0.9, 0.99),
        adam_eps=1e-8,
        fp32_master_optimizer=0,
        quiet_optimizer_log=1,
        vocab_size=int(arch['vocab_size']),
        n_embd=int(arch['n_embd']),
        n_layer=int(arch['n_layer']),
        dim_att=int(arch['n_embd']),
        dim_ffn=int(arch.get('dim_ffn', int(arch['n_embd']) * 4)),
        pre_ffn=0,
        head_qk=0,
        tiny_att_dim=0,
        tiny_att_layer=-999,
    )
    cfg = GRPOConfig(
        num_questions=args.question_batch,
        samples_per_question=args.samples_per_question,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        eval_temperature=args.temperature,
        eval_top_p=args.top_p,
        eval_top_k=args.top_k,
        ppo_epochs=1,
        micro_batch=1,
        rollout_forward_batch=args.rollout_forward_batch,
        lr=1e-7,
        grad_clip=1.0,
        min_tokens=0,
        max_tokens=args.max_new_tokens,
        length_weight=0.0,
        zstd_threshold=2.5,
        zstd_penalty_weight=0.0,
        ngram_penalty=0.0,
        neg_adv_weight=1.0,
        kl_coef=0.0,
        kl_mode='k3_loss',
        time_state_l2=0.0,
        time_state_clamp=0.0,
        log_interval=1,
        save_interval=0,
        eval_interval=0,
        eval_sample_ratio=0.0,
        save_last=False,
        final_full_eval=False,
        hard_buffer_ttl=0,
        hard_buffer_cooldown=0,
        hard_buffer_target_samples=0,
        hard_buffer_group_size=0,
        hard_buffer_extra_lr_scale=0.0,
        hard_buffer_adv_clip=0.0,
        tune_mode='state',
        dapo_dual_clip_c=10.0,
        dapo_dynamic_sampling=False,
        dapo_candidate_questions=0,
        dapo_max_gen_batches=1,
        dapo_min_valid_groups=1,
        dapo_overlong_buffer_tokens=0,
        dapo_overlong_penalty_factor=0.0,
    )

    train_model = RWKVGRPOModel(model_args, cfg, [], [])
    train_model.cuda().to(dtype=rl._rwkv_float_dtype()).eval()
    encode, decode = train_model._build_tokenizer()
    rollout_model = train_model._build_stateful_rollout_model()
    infer = TrainTempBatchInference(rollout_model, train_model, encode, decode, 'cuda', cfg)

    summary = dict(total=0, kept_mixed=0, all0=0, all1=0, correct_samples=0, truncated=0, eod=0, repeat=0, multilingual=0, degenerate=0, no_answer=0)
    t0 = time.time()
    max_prompt_len = int(args.ctx_len) - int(args.max_new_tokens) - 4
    max_prompt_len = max(64, max_prompt_len)

    with mixed_path.open('w', encoding='utf-8') as fm, all_stats_path.open('w', encoding='utf-8') as fs:
        for off in range(0, len(data_slice), args.question_batch):
            batch = data_slice[off:off + args.question_batch]
            prompts = []
            answers = []
            for ex in batch:
                ids = [0] + encode(build_thinking_prompt(ex.get('problem', '')))
                if len(ids) > max_prompt_len:
                    ids = ids[-max_prompt_len:]
                prompts.append(ids)
                answers.append(get_answer(ex))
            comp_tokens, old_logps, texts, truncated, ended_eod = infer.generate_group_parallel(
                prompt_tokens_list=prompts,
                group_size=args.samples_per_question,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                stop_on_token_zero=True,
                stop_on_user=False,
                stop_on_repeat_ngram=False,
                post_trunc_append='',
                post_trunc_max_tokens=0,
            )
            for bi, ex in enumerate(batch):
                rewards=[]; correct=fmt=0
                counters=dict(truncated=0,eod=0,repeat=0,multilingual=0,degenerate=0,no_answer=0)
                sample_stats=[]
                for si in range(args.samples_per_question):
                    idx = bi * args.samples_per_question + si
                    text = texts[idx]
                    r, ok, fok, details = calculate_boxed_think_reward_details(
                        text=text,
                        ground_truth=answers[bi],
                        token_length=len(comp_tokens[idx]),
                        min_tokens=0,
                        max_tokens=args.max_new_tokens,
                        length_weight=0.0,
                        repeat_ngram=False,
                        repeat_penalty=0.0,
                        zstd_threshold=2.5,
                        zstd_penalty_weight=0.0,
                        ended_eod=bool(ended_eod[idx]),
                        truncated=bool(truncated[idx]),
                        overlong_buffer_tokens=0,
                        overlong_penalty_factor=0.0,
                    )
                    rewards.append(float(r))
                    correct += int(bool(ok))
                    fmt += int(bool(fok))
                    counters['truncated'] += int(bool(truncated[idx]))
                    counters['eod'] += int(bool(ended_eod[idx]))
                    counters['repeat'] += int(bool(details.get('degenerate_repeat', False)))
                    counters['multilingual'] += int(bool(details.get('degenerate_multilingual', False)))
                    counters['degenerate'] += int(bool(details.get('degenerate', False)))
                    counters['no_answer'] += int(details.get('extracted_answer') is None)
                    sample_stats.append({'reward': float(r), 'is_correct': bool(ok), 'format': bool(fok), 'truncated': bool(truncated[idx]), 'eod': bool(ended_eod[idx])})
                pos = int(sum(1 for x in rewards if x > 0.5))
                rec = dict(index=args.start + off + bi, pass32_count=pos, pass32_rate=pos/args.samples_per_question, correct_count=correct, format_count=fmt, **counters)
                fs.write(json.dumps(rec, ensure_ascii=False) + '\n'); fs.flush()
                summary['total'] += 1
                summary['correct_samples'] += pos
                for k in counters: summary[k] += counters[k]
                if pos == 0:
                    summary['all0'] += 1
                elif pos == args.samples_per_question:
                    summary['all1'] += 1
                else:
                    summary['kept_mixed'] += 1
                    out_ex = dict(ex)
                    out_ex['pass32_count'] = pos
                    out_ex['pass32_rate'] = pos / args.samples_per_question
                    fm.write(json.dumps(out_ex, ensure_ascii=False) + '\n'); fm.flush()
            done = args.start + off + len(batch)
            elapsed = time.time() - t0
            print(f'[screen] done={done}/{end} kept={summary["kept_mixed"]} all0={summary["all0"]} all1={summary["all1"]} elapsed={elapsed:.1f}s', flush=True)
            summary_path.write_text(json.dumps({**summary, 'elapsed_sec': elapsed, 'input': args.input_jsonl, 'start': args.start, 'end': end, 'samples_per_question': args.samples_per_question}, ensure_ascii=False, indent=2), encoding='utf-8')

    summary_path.write_text(json.dumps({**summary, 'elapsed_sec': time.time()-t0, 'input': args.input_jsonl, 'start': args.start, 'end': end, 'samples_per_question': args.samples_per_question}, ensure_ascii=False, indent=2), encoding='utf-8')
    print('[screen] saved', mixed_path, summary_path, flush=True)

if __name__ == '__main__':
    main()
