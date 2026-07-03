import argparse
import json
import re
import time
from pathlib import Path
from collections import defaultdict, Counter

import torch


def extract_choice(text):
    if text is None:
        return None
    text = str(text)
    m = re.findall(r"\\boxed\s*\{\s*([A-Ja-j])\s*\}", text)
    if m:
        return m[-1].upper()
    m = re.findall(r"^\s*([A-Ja-j])\s*}", text)
    if m:
        return m[-1].upper()
    m = re.findall(r"(?:answer|choice|option)\s*(?:is|:)?\s*\(?\s*([A-Ja-j])\s*\)?", text, flags=re.I)
    if m:
        return m[-1].upper()
    tail = text[-120:]
    m = re.findall(r"(?<![A-Za-z])([A-J])(?![A-Za-z])", tail)
    return m[-1].upper() if m else None


def load_rwkv(project_dir, model_path, tokenizer_path, ctx_len, dtype):
    import os
    import sys
    os.environ['RWKV_HEAD_SIZE_A'] = '64'
    os.environ['RWKV_MY_TESTING'] = 'x070'
    os.environ['RWKV_TRAIN_TYPE'] = 'fullstate'
    os.environ['RWKV_CTXLEN'] = str(int(ctx_len))
    os.environ['FUSED_KERNEL'] = '0'
    os.environ['WKV'] = 'cuda'

    sys.path.insert(0, str(project_dir))
    from main import load_train_model_rwkv7_cuda, normalize_model_arg
    from infer import AlbatrossBatchInference
    from utils import set_seed
    from reference.utils import TRIE_TOKENIZER

    set_seed(123)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True

    tok = TRIE_TOKENIZER(str(tokenizer_path))
    encode = lambda x: tok.encode(x)

    def decode(ids):
        try:
            return tok.decode(ids, utf8_errors="replace")
        except Exception:
            try:
                return tok.decode(ids)
            except Exception:
                try:
                    return tok.decodeBytes(ids).decode("utf-8", errors="replace")
                except Exception:
                    return "".join(chr(int(x) % 256) for x in ids)

    _, pth_path = normalize_model_arg(str(model_path))
    model, _ = load_train_model_rwkv7_cuda(
        pth_path,
        device='cuda',
        ctx_len=ctx_len,
        train_type='fullstate',
        load_dtype=dtype,
    )
    model.eval()
    infer = AlbatrossBatchInference(
        infer_model=None,
        train_model=model,
        encode_fn=encode,
        decode_fn=decode,
        device='cuda',
        cfg=type('Cfg', (), {'tune_mode': 'full', 'rollout_forward_batch': 8})(),
    )
    return encode, decode, infer, model


def read_rows(path):
    rows = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--eval_jsonl', default='/root/RWKV-LM/RWKV7-mmlupro_stem_g1f1p5b_20260624/data_mmlupro_stem_boxprefix/mmlupro_stem_eval_rwkv.jsonl')
    ap.add_argument('--tokenizer', default='/root/RWKV-LM/rwkv_vocab_v20230424.txt')
    ap.add_argument('--project_dir', default='/root/RWKV-LM/RWKV7-mmlupro_stem_g1f1p5b_20260624')
    ap.add_argument('--ctx_len', type=int, default=8192)
    ap.add_argument('--model_dtype', default='bf16')
    ap.add_argument('--max_new_tokens', type=int, default=8)
    ap.add_argument('--temperature', type=float, default=1.0)
    ap.add_argument('--top_p', type=float, default=1.0)
    ap.add_argument('--top_k', type=int, default=0)
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_rows(args.eval_jsonl)
    if args.limit > 0:
        rows = rows[:args.limit]

    print('loading model', flush=True)
    encode, decode, infer, model = load_rwkv(Path(args.project_dir), args.model, args.tokenizer, args.ctx_len, args.model_dtype)

    out_jsonl = out_dir / 'eval.jsonl'
    total = correct = no_parse = trunc_count = 0
    cat_stats = defaultdict(lambda: {'n': 0, 'correct': 0, 'no_parse': 0, 'trunc': 0})
    pred_dist = Counter()
    t0 = time.time()

    with out_jsonl.open('w', encoding='utf-8') as fout:
        chunk_size = 128
        for st in range(0, len(rows), chunk_size):
            chunk = rows[st:st+chunk_size]
            prompt_tokens = []
            for row in chunk:
                ids = encode(row['problem'])
                max_prompt_len = args.ctx_len - args.max_new_tokens - 4
                if len(ids) > max_prompt_len:
                    ids = ids[-max_prompt_len:]
                prompt_tokens.append(ids)
            comp_tokens, _, texts, trunc = infer.generate_group_parallel(
                prompt_tokens_list=prompt_tokens,
                group_size=1,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                stop_on_think_close=False,
                stop_on_user=True,
                stop_on_boxed=True,
                stop_on_repeat_ngram=False,
                presence_penalty=0.0,
                frequency_penalty=0.0,
                alpha_decay=1.0,
            )
            for row, text, toks, tr in zip(chunk, texts, comp_tokens, trunc):
                gt = str(row.get('answer', '')).strip().upper().replace('$', '')
                m = re.search(r'([A-J])', gt)
                gt = m.group(1) if m else gt
                pred = extract_choice(text)
                ok = pred == gt
                cat = row.get('category', 'unknown')
                total += 1
                correct += int(ok)
                no_parse += int(pred is None)
                trunc_count += int(bool(tr))
                cat_stats[cat]['n'] += 1
                cat_stats[cat]['correct'] += int(ok)
                cat_stats[cat]['no_parse'] += int(pred is None)
                cat_stats[cat]['trunc'] += int(bool(tr))
                pred_dist[str(pred)] += 1
                fout.write(json.dumps({
                    'id': row.get('id'),
                    'category': cat,
                    'ground_truth': gt,
                    'pred_extracted': pred,
                    'is_correct': ok,
                    'response': text,
                    'truncated': bool(tr),
                    'gen_len': len(toks),
                    'problem': row.get('problem'),
                }, ensure_ascii=False) + '\n')
            print('progress', min(st + chunk_size, len(rows)), len(rows), 'acc', correct / max(1, total), flush=True)

    subjects = {
        k: {**v, 'acc': v['correct'] / max(1, v['n']), 'no_parse_rate': v['no_parse'] / max(1, v['n']), 'trunc_rate': v['trunc'] / max(1, v['n'])}
        for k, v in sorted(cat_stats.items())
    }
    summary = {
        'benchmark': 'MMLU-Pro STEM heldout text MCQ',
        'model': args.model,
        'n': total,
        'acc': correct / max(1, total),
        'no_parse': no_parse / max(1, total),
        'trunc': trunc_count / max(1, total),
        'elapsed_s': time.time() - t0,
        'subjects': subjects,
        'pred_dist': dict(sorted(pred_dist.items())),
        'out': str(out_jsonl),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    (out_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()