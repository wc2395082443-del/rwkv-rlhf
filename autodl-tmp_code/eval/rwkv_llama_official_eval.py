#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, json, math, argparse, re, time, random
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm

CODE_DIR = '/root/RWKV-LM/RWKV7-statetuning_math500_hb_eval64_v1'
sys.path.insert(0, CODE_DIR)
from main import load_train_model_rwkv7_cuda, load_time_state_only, load_infer_model_albatross
from reference.utils import TRIE_TOKENIZER
from infer import AlbatrossBatchInference
from train import GRPOConfig

DEFAULT_MODEL = '/dev/shm/rwkv_models/rwkv7-g1e-1.5b-20260309-ctx8192.pth'
DEFAULT_TOKENIZER = '/root/RWKV-LM/rwkv_vocab_v20230424.txt'
DEFAULT_MMLU = '/root/RWKV-LM/Albatross/eval/mmlu_test_dataset'
DEFAULT_LAMBADA = '/root/RWKV-LM/Albatross/eval/lambada_test.jsonl'
DEFAULT_GSM8K = '/root/autodl-tmp/data/gsm8k/gsm8k_test_formatted.jsonl'
DEFAULT_MATH500 = '/root/autodl-tmp/data/math500/test.jsonl'

UNSUPPORTED = {
    'ifeval': 'needs official IFEval prompt dataset and instruction-level constraint checker; not safely approximated by exact match',
    'open_rewrite': 'Meta Open-rewrite eval uses task-specific judging not present locally',
    'tldr9': 'TLDR9+ needs summarization judge/metric and dataset not present locally',
    'bfcl_v2': 'tool-calling benchmark; requires BFCL official harness and function-call parser',
    'nexus': 'tool-use benchmark; requires Nexus official environment/evaluator',
    'infinitebench_qa': 'long-context benchmark; dataset/evaluator not present locally',
    'infinitebench_mc': 'long-context benchmark; dataset/evaluator not present locally',
    'nih_multineedle': 'needle-in-haystack benchmark; requires synthetic context generator and exact protocol',
    'agieval_en': 'AGIEval has mixed task formats; official prompt/evaluator not present locally',
    'squad': 'extractive QA F1/EM possible, but official Llama prompt/eval protocol not implemented in this RWKV harness yet',
    'quac': 'dialog QA metric and prompt protocol not implemented in this RWKV harness yet',
    'drop': 'DROP numerical/discrete reasoning metric not implemented in this RWKV harness yet',
    'needle': 'needle-in-haystack protocol not implemented in this RWKV harness yet',
}


def safe_decode(tok, ids):
    try:
        return tok.decode(ids, utf8_errors='replace')
    except TypeError:
        return tok.decode(ids)


def normalize_text(s: str) -> str:
    return re.sub(r'\s+', ' ', str(s)).strip()


def extract_number(text: str) -> Optional[str]:
    text = str(text)
    boxed = re.findall(r'\\boxed\{([^{}]+)\}', text)
    if boxed:
        text = boxed[-1]
    nums = re.findall(r'-?\d+(?:,\d{3})*(?:\.\d+)?', text)
    if not nums:
        return None
    return nums[-1].replace(',', '')


def numeric_equal(a: str, b: str) -> bool:
    ea, eb = extract_number(a), extract_number(b)
    if ea is None or eb is None:
        return normalize_text(a).lower() == normalize_text(b).lower()
    try:
        return abs(float(ea) - float(eb)) < 1e-6
    except Exception:
        return ea == eb


class RWKVEval:
    def __init__(self, args):
        os.environ['RWKV_HEAD_SIZE_A']='64'
        os.environ['RWKV_MY_TESTING']='x070'
        os.environ['RWKV_TRAIN_TYPE']='state'
        os.environ['RWKV_CTXLEN']=str(args.ctx_len)
        os.environ['FUSED_KERNEL']='0'
        os.environ['WKV']='cuda'
        self.args = args
        self.device = 'cuda'
        self.tok = TRIE_TOKENIZER(args.tokenizer)
        self.model, _ = load_train_model_rwkv7_cuda(args.model, self.device, args.ctx_len)
        self.state_loaded = False
        if args.state_init:
            self.state_loaded = load_time_state_only(self.model, args.state_init)
        self.model.eval()
        self._infer_engine = None

    def encode(self, s: str) -> List[int]:
        return self.tok.encode(s)

    def decode(self, ids: List[int]) -> str:
        return safe_decode(self.tok, ids)

    def score_target(self, full_ids: List[int], start: int, max_len: Optional[int]=None) -> Tuple[float, int]:
        max_len = max_len or self.args.ctx_len
        if start >= len(full_ids):
            return -1e30, 0
        ids = list(full_ids)
        if len(ids) > max_len:
            keep_target_len = len(ids) - start
            keep = min(max_len, keep_target_len + max(1, max_len - keep_target_len))
            cut = len(ids) - keep
            ids = ids[cut:]
            start = max(1, start - cut)
        if len(ids) < 2 or start <= 0:
            return -1e30, 0
        inp = torch.tensor([ids[:-1]], dtype=torch.long, device=self.device)
        tgt = torch.tensor([ids[1:]], dtype=torch.long, device=self.device)
        with torch.no_grad():
            logits = self.model(inp).float()
            sl = slice(start - 1, tgt.size(1))
            lp = F.log_softmax(logits[:, sl, :], dim=-1).gather(-1, tgt[:, sl].unsqueeze(-1)).squeeze(-1)
            return float(lp.sum().item()), int(lp.numel())

    def score_completion(self, prompt: str, completion: str, normalize: bool=True) -> float:
        ctx = self.encode(prompt)
        comp = self.encode(completion)
        full = ctx + comp
        score, cnt = self.score_target(full, len(ctx))
        return score / max(cnt, 1) if normalize else score

    def infer_engine(self):
        if self._infer_engine is None:
            base_name = self.args.model[:-4] if self.args.model.endswith('.pth') else self.args.model
            infer_model, _ = load_infer_model_albatross(base_name)
            cfg = GRPOConfig(
                num_questions=1, samples_per_question=1, max_new_tokens=self.args.max_new_tokens,
                temperature=self.args.temperature, top_p=self.args.top_p, top_k=self.args.top_k,
                eval_temperature=self.args.eval_temperature, eval_top_p=self.args.eval_top_p, eval_top_k=self.args.eval_top_k,
                ppo_epochs=1, micro_batch=1, lr=1e-4, grad_clip=1.0,
                min_tokens=0, max_tokens=self.args.max_new_tokens, length_weight=0.0,
                zstd_threshold=2.5, zstd_penalty_weight=0.0, ngram_penalty=0.0,
                answer_judge='auto', neg_adv_weight=1.0, kl_coef=0.0, kl_mode='none',
                time_state_l2=0.0, time_state_clamp=0.0, log_interval=1, save_interval=999999,
                eval_interval=999999, eval_sample_ratio=1.0,
                hard_buffer_ttl=0, hard_buffer_cooldown=0, hard_buffer_target_samples=0,
                hard_buffer_group_size=1, hard_buffer_extra_lr_scale=0.0, hard_buffer_adv_clip=0.0)
            self._infer_engine = AlbatrossBatchInference(infer_model, self.model, self.encode, self.decode, self.device, cfg)
        return self._infer_engine

    def generate_one(self, prompt: str) -> str:
        eng = self.infer_engine()
        comp, _, texts, _ = eng.generate_group_parallel(
            [self.encode(prompt)], group_size=1, max_new_tokens=self.args.max_new_tokens,
            temperature=self.args.eval_temperature, top_p=self.args.eval_top_p, top_k=self.args.eval_top_k,
            stop_on_user=True, stop_on_boxed=False, stop_on_repeat_ngram=True,
            post_trunc_append='', post_trunc_max_tokens=0)
        return texts[0] if texts else self.decode(comp[0])


def load_hf_dataset(name: str, *args, split: str, cache_dir: Optional[str]=None, **kwargs):
    from datasets import load_dataset
    return load_dataset(name, *args, split=split, cache_dir=cache_dir, **kwargs)


def eval_lambada(ev: RWKVEval, path: str, limit: int=0) -> Dict[str, Any]:
    rows = [json.loads(l) for l in open(path, encoding='utf-8') if l.strip()]
    if limit: rows = rows[:limit]
    ok = total = 0; nll = 0.0; ntok = 0
    for r in tqdm(rows, desc='lambada'):
        text = r['text'].rstrip()
        m = re.search(r'\S+\s*$', text)
        if not m: continue
        context, target = text[:m.start()], text[m.start():]
        ctx_ids = ev.encode(context)
        target_ids = ev.encode(target)
        full_ids = ctx_ids + target_ids
        score, cnt = ev.score_target(full_ids, len(ctx_ids))
        nll -= score; ntok += cnt
        pred_ids=[]; cur=list(ctx_ids)
        for _ in range(len(target_ids)):
            inp=torch.tensor([cur[-8191:]], dtype=torch.long, device=ev.device)
            with torch.no_grad():
                logits=ev.model(inp)[:, -1, :].float()
                pred=int(torch.argmax(logits, dim=-1).item())
            pred_ids.append(pred); cur.append(pred)
        ok += int(pred_ids == target_ids); total += 1
    return {'benchmark':'lambada','count':total,'accuracy':ok/total if total else None,'nll':nll/ntok if ntok else None,'ppl':math.exp(nll/ntok) if ntok else None}


def eval_mmlu(ev: RWKVEval, path: str, limit: int=0) -> Dict[str, Any]:
    from datasets import load_from_disk
    ds = load_from_disk(path)
    if limit: ds = ds.select(range(min(limit, len(ds))))
    labels = [' A',' B',' C',' D']
    ok=total=0; bysub={}
    for r in tqdm(ds, desc='mmlu'):
        prompt = r['question'].strip() + '\n' + ''.join(f'{chr(65+i)}. {c}\n' for i,c in enumerate(r['choices'])) + 'Answer:'
        scores = [ev.score_completion(prompt, lab, normalize=True) for lab in labels[:len(r['choices'])]]
        pred = max(range(len(scores)), key=lambda i:scores[i])
        ans = int(r['answer'])
        corr = pred == ans
        ok += int(corr); total += 1
        sub = r.get('subject','unknown')
        a,b = bysub.get(sub,(0,0)); bysub[sub]=(a+int(corr),b+1)
    return {'benchmark':'mmlu','count':total,'accuracy':ok/total if total else None,'subjects':{k:v[0]/v[1] for k,v in bysub.items()}}


def eval_hellaswag(ev: RWKVEval, limit: int=0, split: str='validation') -> Dict[str, Any]:
    ds = load_hf_dataset('hellaswag', split=split, cache_dir=ev.args.cache_dir)
    if limit: ds = ds.select(range(min(limit, len(ds))))
    ok=total=0
    for r in tqdm(ds, desc='hellaswag'):
        ctx = normalize_text((r.get('ctx_a','') + ' ' + r.get('ctx_b','')).strip())
        scores = [ev.score_completion(ctx, ' ' + normalize_text(e), normalize=True) for e in r['endings']]
        pred = max(range(len(scores)), key=lambda i:scores[i])
        ans = int(r['label'])
        ok += int(pred == ans); total += 1
    return {'benchmark':'hellaswag','count':total,'accuracy':ok/total if total else None,'split':split}


def eval_arc_c(ev: RWKVEval, limit: int=0, split: str='test') -> Dict[str, Any]:
    ds = load_hf_dataset('ai2_arc', 'ARC-Challenge', split=split, cache_dir=ev.args.cache_dir)
    if limit: ds = ds.select(range(min(limit, len(ds))))
    ok=total=0
    for r in tqdm(ds, desc='arc_c'):
        q = r['question'].strip()
        labels = list(r['choices']['label']); texts = list(r['choices']['text'])
        prompt = 'Question: ' + q + '\nAnswer:'
        scores = [ev.score_completion(prompt, ' ' + normalize_text(t), normalize=True) for t in texts]
        pred = labels[max(range(len(scores)), key=lambda i:scores[i])]
        ans = str(r['answerKey'])
        ok += int(pred == ans); total += 1
    return {'benchmark':'arc_c','count':total,'accuracy':ok/total if total else None,'split':split}


def eval_gpqa(ev: RWKVEval, limit: int=0, config: str='gpqa_diamond') -> Dict[str, Any]:
    # Public GPQA HF mirrors normally expose columns:
    # Question, Correct Answer, Incorrect Answer 1/2/3.
    last_err = None
    for name in ['Idavidrein/gpqa', 'hendrycks/GPQA']:
        try:
            ds = load_hf_dataset(name, config, split='train', cache_dir=ev.args.cache_dir)
            break
        except Exception as e:
            last_err = e; ds = None
    if ds is None:
        return {'benchmark':'gpqa','status':'unavailable','error':repr(last_err)}
    if limit: ds = ds.select(range(min(limit, len(ds))))
    ok=total=0
    rng = random.Random(1234)
    for idx, r in enumerate(tqdm(ds, desc='gpqa')):
        q = r.get('Question') or r.get('question')
        correct = r.get('Correct Answer') or r.get('correct_answer') or r.get('answer')
        wrongs = [r.get(f'Incorrect Answer {i}') for i in range(1,4)]
        wrongs = [w for w in wrongs if w]
        if not q or not correct or len(wrongs) < 3:
            continue
        choices = [(correct, True)] + [(w, False) for w in wrongs]
        rng.seed(idx); rng.shuffle(choices)
        prompt = 'Question: ' + normalize_text(q) + '\nAnswer:'
        scores = [ev.score_completion(prompt, ' ' + normalize_text(c[0]), normalize=True) for c in choices]
        pred = max(range(len(scores)), key=lambda i:scores[i])
        ok += int(choices[pred][1]); total += 1
    return {'benchmark':'gpqa','config':config,'count':total,'accuracy':ok/total if total else None}


def read_jsonl(path):
    return [json.loads(l) for l in open(path, encoding='utf-8') if l.strip()]


def eval_gsm_jsonl(ev: RWKVEval, path: str, limit: int=0, name: str='gsm8k') -> Dict[str, Any]:
    rows = read_jsonl(path)
    if limit: rows = rows[:limit]
    ok=total=0; examples=[]
    for r in tqdm(rows, desc=name):
        q = r.get('problem') or r.get('question') or r.get('input')
        gt = r.get('ground_truth') or r.get('answer') or r.get('target')
        if q is None or gt is None: continue
        prompt = f'User: Solve the following problem step by step.\n{q}\nAssistant:'
        out = ev.generate_one(prompt)
        corr = numeric_equal(out, gt)
        ok += int(corr); total += 1
        if len(examples) < 3: examples.append({'gt':gt,'pred_num':extract_number(out),'correct':corr})
    return {'benchmark':name,'count':total,'accuracy':ok/total if total else None,'examples':examples,'mode':'generation'}


def eval_mgsm(ev: RWKVEval, limit: int=0, lang: str='en') -> Dict[str, Any]:
    last_err = None
    for name,args in [('juletxara/mgsm',(lang,)), ('google-research-datasets/mgsm',(lang,)), ('mgsm',(lang,))]:
        try:
            ds = load_hf_dataset(name, *args, split='test', cache_dir=ev.args.cache_dir)
            break
        except Exception as e:
            last_err=e; ds=None
    if ds is None:
        return {'benchmark':'mgsm','status':'unavailable','error':repr(last_err)}
    if limit: ds = ds.select(range(min(limit, len(ds))))
    ok=total=0
    for r in tqdm(ds, desc='mgsm'):
        q = r.get('question') or r.get('problem')
        gt = str(r.get('answer') or r.get('target') or '')
        if not q or not gt: continue
        prompt = f'User: Solve the following problem step by step.\n{q}\nAssistant:'
        out = ev.generate_one(prompt)
        ok += int(numeric_equal(out, gt)); total += 1
    return {'benchmark':'mgsm','lang':lang,'count':total,'accuracy':ok/total if total else None,'mode':'generation'}


def unsupported_result(task: str) -> Dict[str, Any]:
    return {'benchmark': task, 'status': 'unsupported', 'reason': UNSUPPORTED.get(task, 'not implemented')}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default=DEFAULT_MODEL)
    ap.add_argument('--state_init', default='')
    ap.add_argument('--tokenizer', default=DEFAULT_TOKENIZER)
    ap.add_argument('--out', default='/dev/shm/rwkv_runs/official_like_eval/results.jsonl')
    ap.add_argument('--tasks', default='mmlu,hellaswag,arc_c,gpqa,mgsm,gsm8k,math500,lambada,ifeval,open_rewrite,tldr9,bfcl_v2,nexus,infinitebench_qa,infinitebench_mc,nih_multineedle')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--ctx_len', type=int, default=8192)
    ap.add_argument('--max_new_tokens', type=int, default=1024)
    ap.add_argument('--temperature', type=float, default=1.0)
    ap.add_argument('--top_p', type=float, default=0.6)
    ap.add_argument('--top_k', type=int, default=500)
    ap.add_argument('--eval_temperature', type=float, default=0.3)
    ap.add_argument('--eval_top_p', type=float, default=0.4)
    ap.add_argument('--eval_top_k', type=int, default=500)
    ap.add_argument('--cache_dir', default='/root/autodl-tmp/hf_datasets')
    ap.add_argument('--mmlu_path', default=DEFAULT_MMLU)
    ap.add_argument('--lambada_path', default=DEFAULT_LAMBADA)
    ap.add_argument('--gsm8k_path', default=DEFAULT_GSM8K)
    ap.add_argument('--math500_path', default=DEFAULT_MATH500)
    args = ap.parse_args()

    tasks = [t.strip() for t in args.tasks.split(',') if t.strip()]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    ev = RWKVEval(args)
    results=[]
    for task in tasks:
        t0=time.time()
        try:
            if task == 'mmlu': r = eval_mmlu(ev, args.mmlu_path, args.limit)
            elif task == 'lambada': r = eval_lambada(ev, args.lambada_path, args.limit)
            elif task == 'hellaswag': r = eval_hellaswag(ev, args.limit)
            elif task == 'arc_c': r = eval_arc_c(ev, args.limit)
            elif task == 'gpqa': r = eval_gpqa(ev, args.limit)
            elif task == 'mgsm': r = eval_mgsm(ev, args.limit)
            elif task == 'gsm8k': r = eval_gsm_jsonl(ev, args.gsm8k_path, args.limit, 'gsm8k')
            elif task == 'math500': r = eval_gsm_jsonl(ev, args.math500_path, args.limit, 'math500')
            else: r = unsupported_result(task)
        except Exception as e:
            r = {'benchmark':task,'status':'error','error':repr(e)}
        r['elapsed_sec']=time.time()-t0
        r['state_init']=args.state_init
        r['state_loaded']=ev.state_loaded
        r['limit']=args.limit
        results.append(r)
        print(json.dumps(r, ensure_ascii=False))
        with open(args.out, 'a', encoding='utf-8') as f:
            f.write(json.dumps(r, ensure_ascii=False)+'\n')
    print('DONE', args.out)

if __name__ == '__main__':
    main()

