#!/usr/bin/env python3
import argparse
import contextlib
import importlib.util
import json
import math
import os
import random
import sys
import time
import types
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
import deepspeed

BASE_DIR = Path('/root/RWKV-LM/RWKV-v7/train_temp')
BASELINE_DIR = Path('/root/RWKV-LM/RWKV7-statetuning_hardbuffer_v1')
os.chdir(BASE_DIR)
for p in [str(BASE_DIR), str(BASELINE_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ.setdefault('RWKV_MY_TESTING', 'x070')
os.environ.setdefault('RWKV_CTXLEN', '8192')
os.environ.setdefault('RWKV_HEAD_SIZE', '64')
os.environ.setdefault('RWKV_FLOAT_MODE', 'bf16')
os.environ.setdefault('RWKV_JIT_ON', '0')
for d in ['/root/miniconda3/bin', '/usr/bin', '/bin']:
    if os.path.isfile(os.path.join(d, 'ninja')) and d not in os.environ.get('PATH', ''):
        os.environ['PATH'] = d + os.pathsep + os.environ.get('PATH', '')
        break

import train_rl_baseline as mod  # noqa: E402

reward_spec = importlib.util.spec_from_file_location('baseline_reward', BASELINE_DIR / 'reward.py')
reward_mod = importlib.util.module_from_spec(reward_spec)
reward_spec.loader.exec_module(reward_mod)


def robust_load_weights(path: str):
    try:
        return mod._torch_load_weights(path)
    except Exception:
        try:
            return torch.load(path, map_location=chr(99)+chr(112)+chr(117), weights_only=False)
        except TypeError:
            return torch.load(path, map_location=chr(99)+chr(112)+chr(117))


def read_jsonl(path: str, max_samples: int = 0) -> List[Dict[str, Any]]:
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if max_samples and len(rows) >= max_samples:
                break
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')


def append_jsonl(path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(obj, ensure_ascii=False) + '\n')


def safe_decode(tok, ids: List[int]) -> str:
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


def build_prompt(problem: str, style: str) -> str:
    p = (problem or '').strip().replace('\r\n', '\n')
    if style == 'albatross_fake_think':
        return f'User: {p}\n\nAssistant: <think></think'
    if style == 'plain':
        return f'User: {p}\n\nAssistant:'
    return mod._baseline_mod.build_prompt(p)


def detect_token_loop(tokens: List[int], min_ngram: int = 8, max_ngram: int = 48, repeats: int = 4) -> Optional[Dict[str, int]]:
    n_tokens = len(tokens)
    if n_tokens < min_ngram * repeats:
        return None
    # Prefer shorter periodic loops, then earliest entry point.
    best = None
    for n in range(min_ngram, min(max_ngram, n_tokens // repeats) + 1):
        limit = n_tokens - n * repeats
        for s in range(0, limit + 1):
            block = tokens[s:s+n]
            ok = True
            for r in range(1, repeats):
                if tokens[s+r*n:s+(r+1)*n] != block:
                    ok = False
                    break
            if ok:
                entry = s + n
                if entry < n_tokens:
                    cand = {'start': s, 'ngram': n, 'entry': entry, 'reject_id': int(tokens[entry])}
                    if best is None or (cand['entry'], cand['ngram']) < (best['entry'], best['ngram']):
                        best = cand
                    break
        if best is not None:
            return best
    return None


def has_token_loop(tokens: List[int], min_ngram: int = 8, max_ngram: int = 48, repeats: int = 4) -> bool:
    return detect_token_loop(tokens, min_ngram=min_ngram, max_ngram=max_ngram, repeats=repeats) is not None


def make_rwkv_args(load_model: str, tokenizer: str, proj_dir: str, precision: str, lr: float, ctx_len: int, grad_cp: int = 1):
    old_argv = sys.argv[:]
    try:
        sys.argv = [
            'antidoom', '--load_model', load_model, '--proj_dir', proj_dir, '--tokenizer', tokenizer,
            '--train_jsonl', '/dev/null', '--ctx_len', str(ctx_len), '--precision', precision,
            '--strategy', 'deepspeed_stage_3_offload', '--micro_bsz', '1', '--grad_cp', str(grad_cp),
            '--lr', str(lr), '--weight_decay', '0.0', '--beta1', '0.9', '--beta2', '0.99',
            '--adam_eps', '1e-8', '--ds_bucket_mb', '64', '--ds_contiguous_gradients', '0',
            '--enable_progress_bar', '0', '--quiet_optimizer_log', '1'
        ]
        args = mod.parse_args()
    finally:
        sys.argv = old_argv
    rwkv_precision = {'32': 'fp32', 32: 'fp32', '16': 'fp16', 16: 'fp16'}.get(args.precision, args.precision)
    os.environ['RWKV_MY_TESTING'] = args.my_testing
    os.environ['RWKV_CTXLEN'] = str(int(args.ctx_len))
    os.environ['RWKV_HEAD_SIZE'] = str(int(args.head_size))
    os.environ['RWKV_FLOAT_MODE'] = rwkv_precision
    os.environ['RWKV_JIT_ON'] = '0'
    args.lr_init = float(lr)
    args.lr_final = float(lr)
    args.betas = (float(args.beta1), float(args.beta2))
    args.train_stage = 0
    args.real_bsz = 1
    sd = mod._normalize_state_dict(robust_load_weights(load_model))
    args.n_layer, args.n_embd, args.vocab_size, args.dim_ffn = mod._infer_arch(sd)
    args.dim_att = args.n_embd
    return args, sd


def load_policy_for_infer(model_path: str, tokenizer_path: str, out_dir: str, precision: str, ctx_len: int):
    args, sd = make_rwkv_args(model_path, tokenizer_path, out_dir, precision, lr=1e-7, ctx_len=ctx_len, grad_cp=0)
    tok = mod.TRIE_TOKENIZER(tokenizer_path)
    policy_model = mod.PaddedRWKV(args)
    policy_model.load_state_dict(sd, strict=True)
    policy_model = mod._cast_ref_model_dtype(policy_model.to('cuda'))
    policy_model.eval()
    for p in policy_model.parameters():
        p.requires_grad = False
    rollout_args = types.SimpleNamespace(MODEL_NAME=str(Path(model_path).with_suffix('')), vocab_size=int(args.vocab_size))
    rollout_model = mod.RWKV_x070(rollout_args)
    rollout_model.eval()
    cfg = types.SimpleNamespace(tune_mode='state', rollout_forward_batch=64)
    infer = mod.TrainTempBatchInference(
        infer_model=rollout_model,
        train_model=policy_model,
        encode_fn=tok.encode,
        decode_fn=lambda ids: safe_decode(tok, ids),
        device='cuda',
        cfg=cfg,
    )
    return args, tok, policy_model, infer


@torch.no_grad()
def top_choices_for_prefix(model, prefix: List[int], reject_id: int, topk: int, ref_topk: int, ban_ids: set) -> Tuple[List[int], List[int], List[float]]:
    max_len = int(os.environ.get('ANTIDOOM_SCORE_MAX_PREFIX', '768'))
    if len(prefix) > max_len:
        prefix = prefix[-max_len:]
    x = torch.tensor([prefix], device='cuda', dtype=torch.long)
    logits = model(x)
    if torch.is_tensor(logits) and logits.dim() == 2:
        logits = logits.unsqueeze(0)
    last = logits[0, -1].float()
    vals, ids = torch.topk(last, k=min(max(ref_topk, topk + 16), last.numel()))
    chosen = []
    for t in ids.detach().cpu().tolist():
        t = int(t)
        if t == int(reject_id) or t in ban_ids:
            continue
        chosen.append(t)
        if len(chosen) >= topk:
            break
    ref_ids = [int(x) for x in ids[:ref_topk].detach().cpu().tolist()]
    ref_vals = [float(x) for x in vals[:ref_topk].detach().cpu().tolist()]
    return chosen, ref_ids, ref_vals


def run_mine(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rwkv_args, tok, policy_model, infer = load_policy_for_infer(args.model, args.tokenizer, str(out_dir), args.precision, args.ctx_len)
    data = read_jsonl(args.dataset, args.max_samples)
    pairs = []
    gen_records = []
    ban_ids = set([0])
    t0 = time.time()
    for start in range(0, len(data), args.chunk_questions):
        batch = data[start:start + args.chunk_questions]
        prompts = []
        prompt_tokens = []
        for ex in batch:
            ps = build_prompt(ex.get('problem', ''), args.prompt_style)
            ids = tok.encode(ps)
            max_prompt_len = int(args.ctx_len) - int(args.max_new_tokens) - 4
            if len(ids) > max_prompt_len:
                ids = ids[-max_prompt_len:]
            prompts.append(ps)
            prompt_tokens.append(ids)
        comp_tokens, logps, comp_texts, truncated = infer.generate_group_parallel(
            prompt_tokens,
            group_size=args.group_size,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            stop_on_think_close=False,
            stop_on_user=True,
            stop_on_boxed=False,
            stop_on_repeat_ngram=False,
            presence_penalty=args.presence_penalty,
            frequency_penalty=args.frequency_penalty,
            post_trunc_append='',
            post_trunc_max_tokens=0,
        )
        for bi, ex in enumerate(batch):
            answer = ex.get('answer', ex.get('solution', ex.get('ground_truth', '')))
            for j in range(args.group_size):
                k = bi * args.group_size + j
                toks = comp_tokens[k]
                text = comp_texts[k]
                loop = detect_token_loop(toks, min_ngram=args.min_ngram, max_ngram=args.max_ngram, repeats=args.repeats)
                reward, is_correct, is_format = reward_mod.calculate_reward(
                    text=text, ground_truth=answer, token_length=len(toks), min_tokens=1,
                    max_tokens=args.max_new_tokens, length_weight=0.0, repeat_ngram=bool(loop),
                    repeat_penalty=0.0, zstd_penalty_weight=0.0,
                )
                gen_records.append({
                    'idx': start + bi, 'sample_idx': j, 'gen_len': len(toks), 'truncated': bool(truncated[k]),
                    'loop': bool(loop), 'is_correct': bool(is_correct), 'is_format_correct': bool(is_format),
                    'response': text[:2000] if args.save_text else '',
                })
                if loop is None:
                    continue
                entry = int(loop['entry'])
                if entry <= 0 or entry >= len(toks):
                    continue
                prefix_full = prompt_tokens[bi] + toks[:entry]
                if len(prefix_full) > args.train_prefix_tokens:
                    prefix_train = prefix_full[-args.train_prefix_tokens:]
                else:
                    prefix_train = prefix_full
                reject_id = int(toks[entry])
                local_ban = set(ban_ids)
                # Avoid replacing the loop-start token with another token from the repeated block when possible.
                s = int(loop['start']); n = int(loop['ngram'])
                local_ban.update(int(x) for x in toks[s:s+n])
                chosen_ids, ref_ids, ref_logits = top_choices_for_prefix(
                    policy_model, prefix_train, reject_id, args.chosen_topk, args.ref_topk, local_ban
                )
                if not chosen_ids:
                    continue
                pairs.append({
                    'idx': start + bi,
                    'sample_idx': j,
                    'problem': ex.get('problem', ''),
                    'answer': answer,
                    'prefix_tokens': [int(x) for x in prefix_train],
                    'reject_id': reject_id,
                    'chosen_ids': chosen_ids,
                    'ref_ids': ref_ids,
                    'ref_logits': ref_logits,
                    'loop': loop,
                    'gen_len': len(toks),
                    'truncated': bool(truncated[k]),
                    'loop_text': safe_decode(tok, toks[max(0, int(loop['start'])-8):min(len(toks), int(loop['entry'])+int(loop['ngram'])*2)])[:1000],
                })
        done = min(len(data), start + len(batch))
        stats = summarize_records(gen_records)
        print(json.dumps({'done': done, 'pairs': len(pairs), **stats, 'elapsed': time.time() - t0}, ensure_ascii=False), flush=True)
        if args.max_pairs and len(pairs) >= args.max_pairs:
            pairs = pairs[:args.max_pairs]
            break
    pair_path = out_dir / 'antidoom_pairs.jsonl'
    gen_path = out_dir / 'mine_generations.jsonl'
    write_jsonl(str(pair_path), pairs)
    write_jsonl(str(gen_path), gen_records)
    summary = {'model': args.model, 'dataset': args.dataset, 'pairs': len(pairs), **summarize_records(gen_records), 'pair_path': str(pair_path), 'gen_path': str(gen_path)}
    with open(out_dir / 'mine_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def summarize_records(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {'samples': 0, 'acc': 0.0, 'loop_rate': 0.0, 'trunc_rate': 0.0, 'avg_len': 0.0}
    return {
        'samples': n,
        'acc': sum(1 for r in rows if r.get('is_correct')) / n,
        'loop_rate': sum(1 for r in rows if r.get('loop')) / n,
        'trunc_rate': sum(1 for r in rows if r.get('truncated')) / n,
        'avg_len': sum(float(r.get('gen_len', 0)) for r in rows) / n,
    }


class AntidoomPairDataset(Dataset):
    def __init__(self, pair_path: str, max_pairs: int = 0, shuffle_seed: int = 42):
        self.rows = read_jsonl(pair_path, max_pairs)
        rng = random.Random(shuffle_seed)
        rng.shuffle(self.rows)
        if not self.rows:
            raise RuntimeError(f'empty antidoom pair dataset: {pair_path}')
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, idx):
        return self.rows[idx % len(self.rows)]


def collate_one(batch):
    return batch[0]


class AntidoomFTPOModel(mod.PaddedRWKV):
    def __init__(self, rwkv_args, load_sd, out_dir: str, total_steps: int, margin: float, pref_weight: float, mse_weight: float, ce_weight: float, grad_clip: float, save_interval: int):
        super().__init__(rwkv_args)
        self.args = rwkv_args
        self.load_state_dict(load_sd, strict=True)
        self.out_dir = out_dir
        self.total_steps = int(total_steps)
        self.margin = float(margin)
        self.pref_weight = float(pref_weight)
        self.mse_weight = float(mse_weight)
        self.ce_weight = float(ce_weight)
        self.grad_clip = float(grad_clip)
        self.save_interval = int(save_interval)
        self.step_idx = 0
        os.makedirs(out_dir, exist_ok=True)

    def configure_optimizers(self):
        if int(getattr(self.args, 'quiet_optimizer_log', 1)) == 1:
            with open(os.devnull, 'w') as devnull, contextlib.redirect_stdout(devnull):
                return super().configure_optimizers()
        return super().configure_optimizers()

    def _save_checkpoint_to(self, out_path: str) -> str:
        state = {}
        for name, param in self.named_parameters():
            with deepspeed.zero.GatheredParameters([param], modifier_rank=None):
                state[name] = param.detach().cpu().clone()
        for name, buf in self.named_buffers():
            state[name] = buf.detach().cpu().clone()
        torch.save(state, out_path)
        return out_path

    def on_train_start(self):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def training_step(self, batch, batch_idx):
        self.step_idx += 1
        prefix = batch['prefix_tokens']
        x = torch.tensor([prefix], device=self.device, dtype=torch.long)
        logits = self(x)
        if torch.is_tensor(logits) and logits.dim() == 2:
            logits = logits.unsqueeze(0)
        last = logits[0, -1].float()
        reject_id = torch.tensor([int(batch['reject_id'])], device=self.device, dtype=torch.long)
        chosen_ids = torch.tensor([int(x) for x in batch['chosen_ids']], device=self.device, dtype=torch.long)
        rej_logit = last.gather(0, reject_id).squeeze(0)
        chosen_logit = torch.logsumexp(last.gather(0, chosen_ids), dim=0) - math.log(max(1, len(batch['chosen_ids'])))
        pref_loss = F.softplus(rej_logit - chosen_logit + self.margin)
        log_probs = F.log_softmax(last, dim=-1)
        ce_loss = -torch.logsumexp(log_probs.gather(0, chosen_ids), dim=0) + math.log(max(1, len(batch['chosen_ids'])))
        if self.mse_weight > 0 and batch.get('ref_ids'):
            ref_ids = torch.tensor([int(x) for x in batch['ref_ids']], device=self.device, dtype=torch.long)
            ref_logits = torch.tensor([float(x) for x in batch['ref_logits']], device=self.device, dtype=torch.float32)
            cur = last.gather(0, ref_ids)
            mse_loss = F.mse_loss(cur, ref_logits)
        else:
            mse_loss = last.sum() * 0.0
        loss = self.pref_weight * pref_loss + self.ce_weight * ce_loss + self.mse_weight * mse_loss
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)
        if self.step_idx % 5 == 0 or self.step_idx == 1:
            peak = torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0.0
            rec = {
                'step': self.step_idx, 'loss': float(loss.detach().cpu()), 'pref_loss': float(pref_loss.detach().cpu()),
                'ce_loss': float(ce_loss.detach().cpu()), 'mse_loss': float(mse_loss.detach().cpu()),
                'rej_logit': float(rej_logit.detach().cpu()), 'chosen_logit': float(chosen_logit.detach().cpu()),
                'prefix_len': len(prefix), 'peak_gb': peak,
            }
            append_jsonl(os.path.join(self.out_dir, 'train_metrics.jsonl'), rec)
            print(json.dumps(rec), flush=True)
        if self.save_interval > 0 and self.step_idx % self.save_interval == 0:
            self._save_checkpoint_to(os.path.join(self.out_dir, f'checkpoint_step_{self.step_idx}.pth'))
        return loss

    def on_train_end(self):
        path = self._save_checkpoint_to(os.path.join(self.out_dir, f'final_step_{self.step_idx}.pth'))
        with open(os.path.join(self.out_dir, 'final_checkpoint.txt'), 'w') as f:
            f.write(path + '\n')
        print(json.dumps({'final_checkpoint': path}), flush=True)


def run_train(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rwkv_args, sd = make_rwkv_args(args.model, args.tokenizer, str(out_dir), args.precision, args.lr, args.ctx_len, grad_cp=args.grad_cp)
    rwkv_args.total_steps = int(args.steps)
    ds = AntidoomPairDataset(args.pairs, max_pairs=args.max_pairs, shuffle_seed=args.seed)
    loader = DataLoader(ds, batch_size=1, shuffle=True, num_workers=0, collate_fn=collate_one)
    model = AntidoomFTPOModel(
        rwkv_args, sd, str(out_dir), args.steps, args.margin, args.pref_weight, args.mse_weight,
        args.ce_weight, args.grad_clip, args.save_interval,
    )
    trainer_precision = {'fp32': '32', 'fp16': '16'}.get(args.precision, args.precision)
    trainer = Trainer(
        accelerator='gpu', devices=1, strategy='deepspeed_stage_3_offload', precision=trainer_precision,
        max_steps=int(args.steps), max_epochs=1000000, logger=False, enable_checkpointing=False,
        num_sanity_val_steps=0, log_every_n_steps=1, enable_progress_bar=False, enable_model_summary=False,
    )
    if 'deepspeed' in str(trainer.strategy):
        trainer.strategy.config['zero_optimization']['allgather_bucket_size'] = int(args.ds_bucket_mb) * 1000 * 1000
        trainer.strategy.config['zero_optimization']['reduce_bucket_size'] = int(args.ds_bucket_mb) * 1000 * 1000
        trainer.strategy.config['zero_optimization']['contiguous_gradients'] = False
    trainer.fit(model, train_dataloaders=loader)


def run_eval(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rwkv_args, tok, policy_model, infer = load_policy_for_infer(args.model, args.tokenizer, str(out_dir), args.precision, args.ctx_len)
    data = read_jsonl(args.dataset, args.max_samples)
    rows = []
    t0 = time.time()
    for start in range(0, len(data), args.chunk_questions):
        batch = data[start:start + args.chunk_questions]
        prompt_tokens = []
        for ex in batch:
            ps = build_prompt(ex.get('problem', ''), args.prompt_style)
            ids = tok.encode(ps)
            max_prompt_len = int(args.ctx_len) - int(args.max_new_tokens) - 4
            if len(ids) > max_prompt_len:
                ids = ids[-max_prompt_len:]
            prompt_tokens.append(ids)
        comp_tokens, logps, comp_texts, truncated = infer.generate_group_parallel(
            prompt_tokens, group_size=args.group_size, max_new_tokens=args.max_new_tokens,
            temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
            stop_on_think_close=False, stop_on_user=True, stop_on_boxed=False,
            stop_on_repeat_ngram=False, presence_penalty=args.presence_penalty,
            frequency_penalty=args.frequency_penalty, post_trunc_append='', post_trunc_max_tokens=0,
        )
        for bi, ex in enumerate(batch):
            answer = ex.get('answer', ex.get('solution', ex.get('ground_truth', '')))
            corrects = 0
            for j in range(args.group_size):
                k = bi * args.group_size + j
                loop = detect_token_loop(comp_tokens[k], min_ngram=args.min_ngram, max_ngram=args.max_ngram, repeats=args.repeats)
                reward, is_correct, is_format = reward_mod.calculate_reward(
                    text=comp_texts[k], ground_truth=answer, token_length=len(comp_tokens[k]), min_tokens=1,
                    max_tokens=args.max_new_tokens, length_weight=0.0, repeat_ngram=bool(loop), repeat_penalty=0.0,
                    zstd_penalty_weight=0.0,
                )
                corrects += int(is_correct)
                rows.append({
                    'idx': start + bi, 'sample_idx': j, 'gen_len': len(comp_tokens[k]), 'truncated': bool(truncated[k]),
                    'loop': bool(loop), 'is_correct': bool(is_correct), 'is_format_correct': bool(is_format),
                    'response': comp_texts[k][:4000] if args.save_text else '',
                })
        done = min(len(data), start + len(batch))
        print(json.dumps({'done': done, **summarize_records(rows), 'elapsed': time.time() - t0}, ensure_ascii=False), flush=True)
    out_path = out_dir / 'eval_antidoom.jsonl'
    write_jsonl(str(out_path), rows)
    summary = {'model': args.model, 'dataset': args.dataset, **summarize_records(rows), 'out_path': str(out_path), 'elapsed': time.time() - t0}
    # pass@k by grouping samples per idx
    by = {}
    for r in rows:
        by.setdefault(int(r['idx']), []).append(r)
    if by:
        summary['first_acc'] = sum(1 for v in by.values() if v[0].get('is_correct')) / len(by)
        summary['pass_at_group'] = sum(1 for v in by.values() if any(x.get('is_correct') for x in v)) / len(by)
    with open(out_dir / 'eval_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def add_common(ap):
    ap.add_argument('--model', required=True)
    ap.add_argument('--tokenizer', default='/root/RWKV-LM/RWKV-v7/rwkv_vocab_v20230424.txt')
    ap.add_argument('--dataset', default='/root/autodl-tmp/Albatross_ref_tmp/faster3a_2605/dataset/MATH500.jsonl')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--max_samples', type=int, default=64)
    ap.add_argument('--group_size', type=int, default=4)
    ap.add_argument('--max_new_tokens', type=int, default=768)
    ap.add_argument('--temperature', type=float, default=1.0)
    ap.add_argument('--top_p', type=float, default=0.28)
    ap.add_argument('--top_k', type=int, default=32)
    ap.add_argument('--presence_penalty', type=float, default=0.0)
    ap.add_argument('--frequency_penalty', type=float, default=0.0)
    ap.add_argument('--prompt_style', choices=['rwkv_boxed', 'albatross_fake_think', 'plain'], default='rwkv_boxed')
    ap.add_argument('--ctx_len', type=int, default=8192)
    ap.add_argument('--precision', default='bf16')
    ap.add_argument('--chunk_questions', type=int, default=4)
    ap.add_argument('--min_ngram', type=int, default=8)
    ap.add_argument('--max_ngram', type=int, default=48)
    ap.add_argument('--repeats', type=int, default=4)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--save_text', type=int, default=0)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)
    pm = sub.add_parser('mine')
    add_common(pm)
    pm.add_argument('--chosen_topk', type=int, default=4)
    pm.add_argument('--ref_topk', type=int, default=64)
    pm.add_argument('--train_prefix_tokens', type=int, default=768)
    pm.add_argument('--max_pairs', type=int, default=128)
    pt = sub.add_parser('train')
    pt.add_argument('--model', required=True)
    pt.add_argument('--tokenizer', default='/root/RWKV-LM/RWKV-v7/rwkv_vocab_v20230424.txt')
    pt.add_argument('--pairs', required=True)
    pt.add_argument('--out_dir', required=True)
    pt.add_argument('--steps', type=int, default=50)
    pt.add_argument('--max_pairs', type=int, default=0)
    pt.add_argument('--lr', type=float, default=5e-7)
    pt.add_argument('--precision', default='bf16')
    pt.add_argument('--ctx_len', type=int, default=8192)
    pt.add_argument('--grad_cp', type=int, default=1)
    pt.add_argument('--margin', type=float, default=0.0)
    pt.add_argument('--pref_weight', type=float, default=1.0)
    pt.add_argument('--ce_weight', type=float, default=0.1)
    pt.add_argument('--mse_weight', type=float, default=0.0005)
    pt.add_argument('--grad_clip', type=float, default=1.0)
    pt.add_argument('--save_interval', type=int, default=0)
    pt.add_argument('--ds_bucket_mb', type=int, default=64)
    pt.add_argument('--seed', type=int, default=42)
    pe = sub.add_parser('eval')
    add_common(pe)
    args = p.parse_args()
    if args.cmd == 'mine':
        run_mine(args)
    elif args.cmd == 'train':
        run_train(args)
    elif args.cmd == 'eval':
        run_eval(args)


if __name__ == '__main__':
    main()
