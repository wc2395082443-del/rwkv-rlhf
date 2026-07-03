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


def make_pair(problem, answer):
    ans = extract_answer(str(answer), str(answer)) or str(answer).strip().split('\n')[-1].strip()
    prompt = SYSTEM + f'User: context is the math problem. Solve it.\nProblem: {problem}\nTurn 1/4:\n\nAssistant:'
    comp = (
        '```repl\n'
        f'final_answer = r"\\\\boxed{{{ans}}}"\n'
        'answer["content"] = final_answer\n'
        'answer["ready"] = True\n'
        '```'
    )
    return prompt, comp


def pad(seqs, pad=0):
    m = max(len(x) for x in seqs)
    out = torch.full((len(seqs), m), pad, dtype=torch.long)
    for i, x in enumerate(seqs):
        out[i, :len(x)] = torch.tensor(x, dtype=torch.long)
    return out


def save_ckpt(model, out_dir, step):
    ck = os.path.join(out_dir, f'ckpt_sft_step{step}.pth')
    torch.save({'step': step, 'model': {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}}, ck)
    print('saved', ck, flush=True)
    return ck


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
    rng = random.Random(args.seed)
    pairs = []
    for ex in data:
        pr, co = make_pair(ex.get('problem', ''), ex.get('answer', ex.get('solution', '')))
        pids, cids = encode(pr), encode(co)
        max_len = args.ctx_len - 8
        if len(pids) + len(cids) > max_len:
            pids = pids[-max(64, max_len - len(cids)):]
        pairs.append((pids, cids))
    print(f'built_pairs={len(pairs)}', flush=True)
    logp = os.path.join(args.out_dir, 'sft_metrics.jsonl')
    t0 = time.time()
    for step in range(1, args.steps + 1):
        batch = rng.sample(pairs, min(args.batch_size, len(pairs)))
        seqs, masks = [], []
        for pids, cids in batch:
            seqs.append(pids + cids)
            masks.append([0] * len(pids) + [1] * len(cids))
        x = pad(seqs).cuda()
        mask = pad(masks).cuda().float()
        inp = x[:, :-1].contiguous(); tgt = x[:, 1:].contiguous(); loss_mask = mask[:, 1:].contiguous()
        logits = model(inp)
        if logits.dim() == 2:
            logits = logits.unsqueeze(0)
        loss_all = F.cross_entropy(logits.float().view(-1, logits.size(-1)), tgt.view(-1), reduction='none').view_as(tgt)
        loss = (loss_all * loss_mask).sum() / loss_mask.sum().clamp_min(1.0)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        rec = {'step': step, 'loss': float(loss.item()), 'tokens': int(loss_mask.sum().item()), 'elapsed': time.time() - t0}
        with open(logp, 'a', encoding='utf-8') as f:
            f.write(json.dumps(rec) + '\n')
        if step % 10 == 0 or step <= 5:
            print(f"[SFT-MIN {step}/{args.steps}] loss={rec['loss']:.4f} tok={rec['tokens']} elapsed={rec['elapsed']:.1f}s", flush=True)
        if args.save_interval > 0 and step % args.save_interval == 0:
            save_ckpt(model, args.out_dir, step)
    ck = save_ckpt(model, args.out_dir, args.steps)
    print('FINAL_CKPT', ck, flush=True)

if __name__ == '__main__':
    main()
