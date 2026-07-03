#!/usr/bin/env python3
import argparse, json, os, random, time
import torch
import torch.nn.functional as F

from main import load_train_model_rwkv7_cuda, normalize_model_arg, unfreeze_all_parameters
from utils import read_jsonl, set_seed
from reward import extract_answer

HEAD_SIZE = 64
SYSTEM = (
    'System: You are a Recursive Language Model (RLM) with a persistent Python REPL. '
    'Solve the task by writing Python code inside ```repl blocks. The REPL has variables/functions: '
    'context (the problem), llm_query(prompt), llm_query_batched(prompts), SHOW_VARS(), and answer. '
    'To submit the final answer, execute: answer["content"] = "\\\\boxed{...}"; answer["ready"] = True. '
    'Do not answer outside the REPL. Use at least one ```repl block.\n\n'
)


def _clean_text(s: str, limit: int = 2200) -> str:
    s = str(s or '').replace('\r\n', '\n').replace('\r', '\n').strip()
    s = s.replace('```', "'''")
    if len(s) > limit:
        s = s[:limit].rstrip() + ' ...'
    return s


def _answer_text(answer: str) -> str:
    ans = extract_answer(str(answer), str(answer))
    if ans is None or str(ans).strip() == '':
        ans = str(answer).strip().split('\n')[-1].strip()
    return str(ans).strip()


def _prompt(problem: str) -> str:
    return SYSTEM + f'User: context is the math problem. Solve it.\nProblem: {problem}\nTurn 1/4:\n\nAssistant:'


def make_minimal_pair(problem, answer):
    ans = _answer_text(answer)
    comp = (
        '```repl\n'
        f'final_answer = r"\\\\boxed{{{ans}}}"\n'
        'answer["content"] = final_answer\n'
        'answer["ready"] = True\n'
        '```'
    )
    return _prompt(problem), comp


def make_comment_solution_pair(problem, answer, solution=''):
    ans = _answer_text(answer)
    sol = _clean_text(solution or answer)
    comment_lines = []
    for line in sol.split('\n'):
        line = line.rstrip()
        if not line:
            comment_lines.append('#')
        else:
            comment_lines.append('# ' + line)
    code = (
        '# Reasoning trace from the supervised solution.\n'
        + '\n'.join(comment_lines)
        + '\n'
        + f'final_answer = r"\\\\boxed{{{ans}}}"\n'
        + 'answer["content"] = final_answer\n'
        + 'answer["ready"] = True\n'
    )
    comp = '```repl\n' + code + '```'
    return _prompt(problem), comp


def pad(seqs, pad_id=0):
    m = max(len(x) for x in seqs)
    out = torch.full((len(seqs), m), pad_id, dtype=torch.long)
    for i, x in enumerate(seqs):
        out[i, :len(x)] = torch.tensor(x, dtype=torch.long)
    return out


def save_ckpt(model, out_dir, step):
    ck = os.path.join(out_dir, f'ckpt_sft_step{step}.pth')
    torch.save({'step': step, 'model': {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}}, ck)
    print('saved', ck, flush=True)
    return ck


def build_pairs(data, encode, ctx_len, minimal_ratio, solution_limit):
    pairs = []
    dropped = 0
    max_len = ctx_len - 8
    for ex in data:
        problem = ex.get('problem', '')
        answer = ex.get('answer', ex.get('solution', ''))
        solution = _clean_text(ex.get('solution', answer), solution_limit)
        variants = []
        variants.append(('minimal',) + make_minimal_pair(problem, answer))
        variants.append(('comment_solution',) + make_comment_solution_pair(problem, answer, solution))
        for kind, pr, co in variants:
            pids, cids = encode(pr), encode(co)
            if len(cids) >= max_len - 64:
                dropped += 1
                continue
            if len(pids) + len(cids) > max_len:
                pids = pids[-max(64, max_len - len(cids)):]
            weight = minimal_ratio if kind == 'minimal' else max(0.0, 1.0 - minimal_ratio)
            if weight > 0:
                pairs.append({'kind': kind, 'pids': pids, 'cids': cids, 'weight': weight})
    if not pairs:
        raise RuntimeError('no SFT pairs built')
    return pairs, dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--tokenizer', required=True)
    ap.add_argument('--train_jsonl', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--ctx_len', type=int, default=8192)
    ap.add_argument('--model_dtype', default='bf16')
    ap.add_argument('--steps', type=int, default=500)
    ap.add_argument('--batch_size', type=int, default=1)
    ap.add_argument('--lr', type=float, default=1e-6)
    ap.add_argument('--max_samples', type=int, default=12000)
    ap.add_argument('--minimal_ratio', type=float, default=0.5)
    ap.add_argument('--solution_limit', type=int, default=2200)
    ap.add_argument('--save_interval', type=int, default=0)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    os.environ['RWKV_HEAD_SIZE_A'] = str(HEAD_SIZE)
    os.environ['RWKV_MY_TESTING'] = 'x070'
    os.environ['RWKV_TRAIN_TYPE'] = 'fullstate'
    os.environ['RWKV_CTXLEN'] = str(args.ctx_len)
    os.environ['FUSED_KERNEL'] = '0'
    os.environ['WKV'] = 'cuda'

    from reference.utils import TRIE_TOKENIZER
    tok = TRIE_TOKENIZER(args.tokenizer)
    encode = lambda s: tok.encode(s)

    _, pth = normalize_model_arg(args.model)
    model, _ = load_train_model_rwkv7_cuda(pth, 'cuda', args.ctx_len, 'fullstate', args.model_dtype)
    unfreeze_all_parameters(model)
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, betas=(0.9, 0.95), eps=1e-8)

    data = read_jsonl(args.train_jsonl, max_samples=args.max_samples)
    pairs, dropped = build_pairs(data, encode, args.ctx_len, args.minimal_ratio, args.solution_limit)
    weights = [p['weight'] for p in pairs]
    n_min = sum(1 for p in pairs if p['kind'] == 'minimal')
    n_sol = sum(1 for p in pairs if p['kind'] == 'comment_solution')
    print(f'built_pairs={len(pairs)} minimal={n_min} comment_solution={n_sol} dropped={dropped} minimal_ratio={args.minimal_ratio}', flush=True)

    rng = random.Random(args.seed)
    logp = os.path.join(args.out_dir, 'sft_metrics.jsonl')
    t0 = time.time()
    for step in range(1, args.steps + 1):
        batch = rng.choices(pairs, weights=weights, k=min(args.batch_size, len(pairs)))
        seqs, masks = [], []
        kind_counts = {}
        for item in batch:
            pids, cids = item['pids'], item['cids']
            seqs.append(pids + cids)
            masks.append([0] * len(pids) + [1] * len(cids))
            kind_counts[item['kind']] = kind_counts.get(item['kind'], 0) + 1
        x = pad(seqs).cuda()
        mask = pad(masks).cuda().float()
        inp = x[:, :-1].contiguous()
        tgt = x[:, 1:].contiguous()
        loss_mask = mask[:, 1:].contiguous()
        logits = model(inp)
        if logits.dim() == 2:
            logits = logits.unsqueeze(0)
        loss_all = F.cross_entropy(logits.float().view(-1, logits.size(-1)), tgt.view(-1), reduction='none').view_as(tgt)
        loss = (loss_all * loss_mask).sum() / loss_mask.sum().clamp_min(1.0)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        rec = {
            'step': step,
            'loss': float(loss.item()),
            'tokens': int(loss_mask.sum().item()),
            'kind_counts': kind_counts,
            'elapsed': time.time() - t0,
        }
        with open(logp, 'a', encoding='utf-8') as f:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
        if step <= 5 or step % 10 == 0:
            print(f"[SFT-MIX {step}/{args.steps}] loss={rec['loss']:.4f} tok={rec['tokens']} kinds={kind_counts} elapsed={rec['elapsed']:.1f}s", flush=True)
        if args.save_interval > 0 and step % args.save_interval == 0:
            save_ckpt(model, args.out_dir, step)
    ck = save_ckpt(model, args.out_dir, args.steps)
    print('FINAL_CKPT', ck, flush=True)


if __name__ == '__main__':
    main()
