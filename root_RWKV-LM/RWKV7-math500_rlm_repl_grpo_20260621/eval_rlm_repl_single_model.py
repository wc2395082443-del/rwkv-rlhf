#!/usr/bin/env python3
import argparse, json, os, random, time
from pathlib import Path
from typing import Any, Dict, List
import torch

from main import load_train_model_rwkv7_cuda, normalize_model_arg
from infer import AlbatrossBatchInference
from rlm_repl_runtime import LocalRLMRepl, find_repl_blocks, format_repl_outputs
from reward import calculate_reward_details
from utils import read_jsonl, append_jsonl

HEAD_SIZE = 64


def rlm_system_prompt() -> str:
    return (
        'You are a Recursive Language Model (RLM) with a persistent Python REPL. '
        'Solve the task by writing Python code inside ```repl blocks. The REPL has variables/functions: '
        'context (the problem), llm_query(prompt), llm_query_batched(prompts), SHOW_VARS(), and answer. '
        'To submit the final answer, execute: answer["content"] = "\\\\boxed{...}"; answer["ready"] = True. '
        'Do not answer outside the REPL. Use at least one ```repl block.'
    )


def build_prompt(encode, history: List[Dict[str, str]], ctx_len: int, max_new: int) -> List[int]:
    parts = ['System: ' + rlm_system_prompt()]
    for m in history:
        role = m.get('role', 'user')
        parts.append(('Assistant: ' if role == 'assistant' else 'User: ') + m.get('content', ''))
    parts.append('Assistant:')
    ids = encode('\n\n'.join(parts))
    max_prompt = max(64, ctx_len - max_new - 4)
    return ids[-max_prompt:]


def get_answer(ex: Dict[str, Any]) -> str:
    return str(ex.get('answer', ex.get('solution', '')))


def has_repeat(text: str, n: int = 16, repeat: int = 5) -> bool:
    import re
    toks = re.findall(r'\w+|[^\w\s]', text or '')
    if len(toks) < n * repeat:
        return False
    seen = {}
    for i in range(len(toks) - n + 1):
        ng = tuple(toks[i:i+n]); seen[ng] = seen.get(ng, 0) + 1
        if seen[ng] >= repeat:
            return True
    return False


def rollout_one(ex, infer, encode, ctx_len, max_new, temperature, top_p, top_k, max_turns=4):
    problem = ex.get('problem', '')
    hist = [{'role':'user','content':f'context is the math problem. Solve it.\nProblem: {problem}\nTurn 1/{max_turns}:'}]
    repl = LocalRLMRepl(context=problem, llm_query=lambda p: 'Sub-LLM unavailable in local eval; solve directly and return a boxed answer.')
    segments=[]; final=None; repl_calls=0; trunc=False
    for turn in range(max_turns):
        prompt = build_prompt(encode, hist, ctx_len, max_new)
        comp_tokens, old_logps, texts, truncs = infer.generate_group_parallel(
            [prompt], group_size=1, max_new_tokens=max_new,
            temperature=temperature, top_p=top_p, top_k=top_k,
            stop_on_think_close=False, stop_on_user=True, stop_on_boxed=False,
            stop_on_repeat_ngram=True, presence_penalty=0.0, frequency_penalty=0.0,
            alpha_decay=1.0, stop_strings=['\n```'],
        )
        text = texts[0]
        trunc = bool(trunc or truncs[0])
        hist.append({'role':'assistant','content':text})
        segments.append({'prompt_tokens':prompt, 'comp_tokens':comp_tokens[0], 'text':text})
        outputs=[]
        for code in find_repl_blocks(text):
            out = repl.execute(code)
            outputs.append(out); repl_calls += 1
            if out.get('final_answer') is not None:
                final = out.get('final_answer')
                break
        if outputs:
            hist.append({'role':'user','content':format_repl_outputs(outputs)})
        else:
            hist.append({'role':'user','content':'No REPL code block found. Use a ```repl block and set answer when done.'})
        if final is not None:
            break
        if turn + 1 < max_turns:
            hist.append({'role':'user','content':f'Turn {turn+2}/{max_turns}: continue. Use REPL and submit answer if ready.'})
    if final is None:
        final = next((m.get('content','') for m in reversed(hist) if m.get('role') == 'assistant'), '')
    total_len = sum(len(s['comp_tokens']) for s in segments)
    return {'problem':problem,'answer':get_answer(ex),'response':str(final),'history':hist,'truncated':trunc,'gen_len':total_len,'repl_calls':repl_calls}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model', required=True); ap.add_argument('--tokenizer', required=True); ap.add_argument('--eval_jsonl', required=True); ap.add_argument('--out_dir', required=True)
    ap.add_argument('--ctx_len', type=int, default=8192); ap.add_argument('--model_dtype', default='bf16')
    ap.add_argument('--limit', type=int, default=100); ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--max_new_tokens', type=int, default=384); ap.add_argument('--temperature', type=float, default=0.5); ap.add_argument('--top_p', type=float, default=0.28); ap.add_argument('--top_k', type=int, default=32)
    args=ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    os.environ['RWKV_HEAD_SIZE_A']=str(HEAD_SIZE); os.environ['RWKV_MY_TESTING']='x070'; os.environ['RWKV_TRAIN_TYPE']='fullstate'; os.environ['RWKV_CTXLEN']=str(args.ctx_len); os.environ['FUSED_KERNEL']='0'; os.environ['WKV']='cuda'
    from reference.utils import TRIE_TOKENIZER
    tok=TRIE_TOKENIZER(args.tokenizer); encode=lambda s: tok.encode(s); decode=lambda ids: tok.decode(ids)
    _, pth=normalize_model_arg(args.model)
    model,_=load_train_model_rwkv7_cuda(pth,'cuda',args.ctx_len,'fullstate',args.model_dtype)
    model.eval()
    for p in model.parameters(): p.requires_grad=False
    infer=AlbatrossBatchInference(None, model, encode, decode, 'cuda', type('Cfg',(),{'tune_mode':'full','rollout_forward_batch':1})())
    data=read_jsonl(args.eval_jsonl)
    rng=random.Random(args.seed)
    if args.limit and args.limit < len(data):
        idxs=rng.sample(range(len(data)), args.limit)
        data=[data[i] for i in idxs]
    out_eval=os.path.join(args.out_dir,'eval.jsonl'); out_metrics=os.path.join(args.out_dir,'metrics.jsonl')
    t0=time.time(); total=correct=trunc=fmt=noans=rep=repl_calls=total_len=0
    for i,ex in enumerate(data,1):
        item=rollout_one(ex,infer,encode,args.ctx_len,args.max_new_tokens,args.temperature,args.top_p,args.top_k)
        repeat=has_repeat(item['response'])
        reward,is_corr,is_fmt,details=calculate_reward_details(
            text=item['response'], ground_truth=item['answer'], token_length=item['gen_len'],
            min_tokens=1, max_tokens=args.max_new_tokens, length_weight=0.005,
            repeat_ngram=repeat, repeat_penalty=0.05, zstd_threshold=2.5, zstd_penalty_weight=0.0,
            reward_mode='trl_doc')
        rec={**item,'reward':reward,'is_correct':is_corr,'is_format_correct':is_fmt,'reward_details':details,'repeat_16gram_5':repeat,'idx':i}
        append_jsonl(out_eval, rec)
        total+=1; correct+=int(bool(is_corr)); trunc+=int(bool(item['truncated'])); fmt+=int(bool(is_fmt)); noans+=int(not details.get('extracted_answer')); rep+=int(bool(repeat)); repl_calls+=int(item['repl_calls']); total_len+=int(item['gen_len'])
        if i % 10 == 0:
            print(f'[eval {i}/{len(data)}] acc={correct/total:.3f} trunc={trunc/total:.3f} fmt={fmt/total:.3f} repl={repl_calls/total:.2f}', flush=True)
    metrics={'count':total,'accuracy':correct/max(1,total),'trunc_rate':trunc/max(1,total),'format_rate':fmt/max(1,total),'no_answer_rate':noans/max(1,total),'repeat_rate':rep/max(1,total),'avg_repl_calls':repl_calls/max(1,total),'avg_len':total_len/max(1,total),'eval_time':time.time()-t0,'model':args.model,'temperature':args.temperature,'top_p':args.top_p,'top_k':args.top_k,'max_new_tokens':args.max_new_tokens}
    append_jsonl(out_metrics, metrics)
    print('METRICS', json.dumps(metrics, ensure_ascii=False), flush=True)

if __name__ == '__main__':
    main()
