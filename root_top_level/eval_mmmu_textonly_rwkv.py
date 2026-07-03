import argparse
import ast
import json
import re
import time
from pathlib import Path
from collections import defaultdict

import torch
from datasets import load_dataset


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


def parse_options(x):
    if x is None:
        return []
    if isinstance(x, list):
        return [str(v) for v in x]
    s = str(x).strip()
    if not s:
        return []
    try:
        v = ast.literal_eval(s)
        if isinstance(v, (list, tuple)):
            return [str(z) for z in v]
    except Exception:
        pass
    return [s]


def normalize_answer(ans, options):
    if ans is None:
        return None
    s = str(ans).strip()
    if re.fullmatch(r"[A-Ja-j]", s):
        return s.upper()
    for i, opt in enumerate(options):
        if s == str(opt).strip():
            return chr(ord('A') + i)
    return None


def build_prompt(row):
    question = str(row.get('question', '')).strip()
    options = parse_options(row.get('options'))
    lines = ["User: " + question]
    if options:
        lines.append("Answer Choices:")
        for i, opt in enumerate(options):
            lines.append(f"({chr(ord('A') + i)}) {opt}")
    lines.append("\nChoose the correct option. Output only the final answer in \\boxed{} (for example, \\boxed{A}).")
    lines.append("\nAssistant: \\boxed{")
    return "\n".join(lines), options


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--tokenizer', default='/root/RWKV-LM/rwkv_vocab_v20230424.txt')
    ap.add_argument('--project_dir', default='/root/RWKV-LM/RWKV7-mmlupro_stem_g1f1p5b_20260624')
    ap.add_argument('--ctx_len', type=int, default=8192)
    ap.add_argument('--model_dtype', default='bf16')
    ap.add_argument('--max_new_tokens', type=int, default=8)
    ap.add_argument('--temperature', type=float, default=1.0)
    ap.add_argument('--top_p', type=float, default=1.0)
    ap.add_argument('--top_k', type=int, default=0)
    ap.add_argument('--limit_per_subject', type=int, default=0)
    ap.add_argument('--subjects', default='')
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    subjects = [s.strip() for s in args.subjects.split(',') if s.strip()]
    if not subjects:
        subjects = [
            'Accounting','Agriculture','Architecture_and_Engineering','Art','Art_Theory',
            'Basic_Medical_Science','Biology','Chemistry','Clinical_Medicine','Computer_Science',
            'Design','Diagnostics_and_Laboratory_Medicine','Economics','Electronics','Energy_and_Power',
            'Finance','Geography','History','Literature','Manage','Marketing','Materials','Math',
            'Mechanical_Engineering','Music','Pharmacy','Physics','Psychology','Public_Health','Sociology'
        ]

    print('loading model', flush=True)
    encode, decode, infer, model = load_rwkv(Path(args.project_dir), args.model, args.tokenizer, args.ctx_len, args.model_dtype)

    total = correct = no_parse = skipped = trunc_count = 0
    subj_stats = defaultdict(lambda: {'n': 0, 'correct': 0, 'no_parse': 0, 'trunc': 0})
    out_jsonl = out_dir / 'mmmu_textonly_eval.jsonl'
    t0 = time.time()

    with out_jsonl.open('w', encoding='utf-8') as fout:
        for subj in subjects:
            print('load subject', subj, flush=True)
            try:
                ds = load_dataset('MMMU/MMMU', subj, split='validation', streaming=True)
            except Exception as e:
                print('subject_failed', subj, repr(e), flush=True)
                continue
            if args.limit_per_subject > 0:
                ds = ds.select(range(min(args.limit_per_subject, len(ds))))
            rows = []
            for row in ds:
                opts = parse_options(row.get('options'))
                gt = normalize_answer(row.get('answer'), opts)
                if not opts or gt is None:
                    skipped += 1
                    continue
                prompt, opts = build_prompt(row)
                rows.append((row, prompt, opts, gt))
            print('subject_rows', subj, len(rows), 'skipped_so_far', skipped, flush=True)

            chunk_size = 128
            for st in range(0, len(rows), chunk_size):
                chunk = rows[st:st+chunk_size]
                prompt_tokens = []
                for _, prompt, _, _ in chunk:
                    ids = encode(prompt)
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
                for (row, prompt, opts, gt), text, toks, tr in zip(chunk, texts, comp_tokens, trunc):
                    pred = extract_choice(text)
                    ok = pred == gt
                    total += 1
                    correct += int(ok)
                    no_parse += int(pred is None)
                    trunc_count += int(bool(tr))
                    subj_stats[subj]['n'] += 1
                    subj_stats[subj]['correct'] += int(ok)
                    subj_stats[subj]['no_parse'] += int(pred is None)
                    subj_stats[subj]['trunc'] += int(bool(tr))
                    fout.write(json.dumps({
                        'id': row.get('id'),
                        'subject': subj,
                        'question_type': row.get('question_type'),
                        'answer': gt,
                        'pred': pred,
                        'correct': ok,
                        'response': text,
                        'truncated': bool(tr),
                        'gen_len': len(toks),
                        'question': row.get('question'),
                        'options': opts,
                    }, ensure_ascii=False) + '\n')
            cur = correct / max(1, total)
            print('subject_done', subj, 'total', total, 'acc', cur, flush=True)

    by_subject = {
        k: {**v, 'acc': v['correct'] / max(1, v['n']), 'no_parse_rate': v['no_parse'] / max(1, v['n']), 'trunc_rate': v['trunc'] / max(1, v['n'])}
        for k, v in sorted(subj_stats.items())
    }
    summary = {
        'benchmark': 'MMMU validation text-only MCQ (images ignored; non-official)',
        'model': args.model,
        'n': total,
        'skipped': skipped,
        'acc': correct / max(1, total),
        'no_parse': no_parse / max(1, total),
        'trunc': trunc_count / max(1, total),
        'elapsed_s': time.time() - t0,
        'subjects': by_subject,
        'out': str(out_jsonl),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    (out_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()