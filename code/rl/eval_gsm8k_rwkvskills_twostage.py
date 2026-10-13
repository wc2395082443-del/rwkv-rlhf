#!/usr/bin/env python3
import os, json, time, argparse, types, re
from pathlib import Path
import torch

os.environ.setdefault('RWKV_MY_TESTING', 'x070')
os.environ.setdefault('RWKV_CTXLEN', '16384')
os.environ.setdefault('RWKV_HEAD_SIZE', '64')
os.environ.setdefault('RWKV_FLOAT_MODE', 'bf16')
os.environ.setdefault('RWKV_JIT_ON', '0')

import train_rl_thinking_cot_strict_boxedprompt_schedopt_earlystop as T

MATH_PROMPT = (
    "User?Solve the problem using one clean solution path. Keep each step concise. "
    "Put the complete final answer inside \\(\\boxed{{...}}\\), preserving every required component such as "
    "ordered pairs, fractions, degree symbols, units,names, or expressions. Do not return only the first component. "
    "Do not use a choice letter unless the problem asks for a letter. Do not restart, repeat, enumerate alternative methods, "
    "or re-check solved cases. If the reasoning starts to loop, stop and give the best final answer.\n"
    "{q}?Bot?<think>"
)
FINAL_SUFFIX = "</think>\nTherefore, the answer is \\(\\boxed{"

def read_jsonl(path, limit=0):
    rows=[]
    with open(path,'r',encoding='utf-8') as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if limit and len(rows)>=limit: break
    return rows

def get_answer(row):
    s = str(row.get('solution') or row.get('answer') or row.get('target') or row.get('ground_truth') or '')
    m = re.search(r'\\boxed\{([^{}]+)\}', s)
    return m.group(1).strip() if m else s

def make_args(load_model, tokenizer, ctx_len):
    return types.SimpleNamespace(load_model=load_model, tokenizer=tokenizer, ctx_len=ctx_len, head_size=64, my_testing='x070', precision='bf16', weight_decay=0.0, beta1=0.9, beta2=0.99, adam_eps=1e-8, fp32_master_optimizer=0, lr_init=0.0, lr_final=0.0, betas=(0.9,0.99), train_stage=0, grad_cp=0)

def build_model(model_path, tokenizer, ctx_len, max_new):
    sd=T._normalize_state_dict(T._torch_load_weights(model_path))
    n_layer,n_embd,vocab_size,dim_ffn=T._infer_arch(sd)
    margs=make_args(model_path, tokenizer, ctx_len)
    margs.n_layer=n_layer; margs.n_embd=n_embd; margs.vocab_size=vocab_size; margs.dim_ffn=dim_ffn; margs.dim_att=n_embd
    cfg=T.GRPOConfig(max_new_tokens=max_new, eval_temperature=0.8, eval_top_p=0.35, eval_top_k=40)
    cfg.max_tokens=max_new; cfg.min_tokens=16; cfg.rollout_forward_batch=64
    model=T.RWKVGRPOModel(args=margs, rl_cfg=cfg, train_data=[{}], test_data=[{}], full_test_data=[{}])
    model=model.to('cuda', dtype=T._rwkv_float_dtype()).eval()
    encode,decode=model._build_tokenizer(); model.prepare_stateful_rollout()
    infer=T.TrainTempBatchInference(model, model, encode, decode, 'cuda', cfg)
    return model, infer, encode, decode, cfg

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model', required=True); ap.add_argument('--tokenizer', required=True); ap.add_argument('--eval_jsonl', required=True); ap.add_argument('--out_dir', required=True)
    ap.add_argument('--limit', type=int, default=0); ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--cot_max_tokens', type=int, default=4096); ap.add_argument('--final_max_tokens', type=int, default=128)
    ap.add_argument('--cot_temperature', type=float, default=0.8); ap.add_argument('--cot_top_p', type=float, default=0.35); ap.add_argument('--cot_top_k', type=int, default=40)
    ap.add_argument('--final_temperature', type=float, default=0.8); ap.add_argument('--final_top_p', type=float, default=0.30); ap.add_argument('--final_top_k', type=int, default=8)
    args=ap.parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    torch.manual_seed(424242); torch.cuda.manual_seed_all(424242); torch.backends.cudnn.benchmark=True; torch.backends.cuda.matmul.allow_tf32=True
    data=read_jsonl(args.eval_jsonl, args.limit)
    ctx_len=max(16384, args.cot_max_tokens + args.final_max_tokens + 4096)
    model,infer,encode,decode,cfg=build_model(args.model,args.tokenizer,ctx_len,args.cot_max_tokens)
    total=loose=stage1_eod=stage1_trunc=stage2_eod=stage2_trunc=0; toks1=toks2=0
    t0=time.time(); resp_path=Path(args.out_dir)/'responses.jsonl'
    with resp_path.open('w',encoding='utf-8') as wf, torch.no_grad():
        for start in range(0,len(data),args.batch):
            rows=data[start:start+args.batch]
            prompts=[]
            for row in rows:
                p=MATH_PROMPT.format(q=row.get('problem',''))
                ids=[0]+encode(p)
                max_prompt=ctx_len-args.cot_max_tokens-args.final_max_tokens-16
                if len(ids)>max_prompt: ids=ids[-max(64,max_prompt):]
                prompts.append(ids)
            c1,_,t1,tr1,e1=infer.generate_group_parallel(prompts,1,args.cot_max_tokens,args.cot_temperature,args.cot_top_p,args.cot_top_k,True,False,True,'',0)
            final_prompts=[]
            for ids,txt in zip(prompts,t1):
                final_prompts.append(ids + encode(txt + FINAL_SUFFIX))
            c2,_,t2,tr2,e2=infer.generate_group_parallel(final_prompts,1,args.final_max_tokens,args.final_temperature,args.final_top_p,args.final_top_k,True,False,True,'',0)
            for i,row in enumerate(rows):
                full=t1[i]+FINAL_SUFFIX+t2[i]
                reward,is_correct,is_fmt,details=T.calculate_boxed_think_reward_details(text=full, ground_truth=get_answer(row), token_length=len(c1[i])+len(c2[i]), min_tokens=16, max_tokens=args.cot_max_tokens+args.final_max_tokens, length_weight=0.0, repeat_ngram=False, repeat_penalty=0.0, zstd_penalty_weight=0.0, ended_eod=bool(e2[i]), truncated=bool(tr1[i] or tr2[i]), overlong_buffer_tokens=0, overlong_penalty_factor=0.0)
                total+=1; loose+=int(is_correct); stage1_eod+=int(e1[i]); stage1_trunc+=int(tr1[i]); stage2_eod+=int(e2[i]); stage2_trunc+=int(tr2[i]); toks1+=len(c1[i]); toks2+=len(c2[i])
                wf.write(json.dumps({'index': total-1, 'problem': row.get('problem',''), 'ground_truth': get_answer(row), 'stage1_cot': t1[i], 'stage2_answer_completion': t2[i], 'full_scoring_text': full, 'is_correct': bool(is_correct), 'stage1_eod': bool(e1[i]), 'stage1_truncated': bool(tr1[i]), 'stage2_eod': bool(e2[i]), 'stage2_truncated': bool(tr2[i]), 'reward_details': details}, ensure_ascii=False)+'\n')
            print(f'progress {total}/{len(data)} acc={loose/max(1,total):.4f} s1_trunc={stage1_trunc/max(1,total):.3f} s2_trunc={stage2_trunc/max(1,total):.3f}', flush=True)
    summary={'model': args.model, 'eval_jsonl': args.eval_jsonl, 'num_tasks': total, 'accuracy': loose/max(1,total), 'stage1_eod_rate': stage1_eod/max(1,total), 'stage1_trunc_rate': stage1_trunc/max(1,total), 'stage2_eod_rate': stage2_eod/max(1,total), 'stage2_trunc_rate': stage2_trunc/max(1,total), 'mean_stage1_tokens': toks1/max(1,total), 'mean_stage2_tokens': toks2/max(1,total), 'prompt':'rwkv-skills two-stage math prompt', 'elapsed_sec': time.time()-t0, 'responses_path': str(resp_path)}
    (Path(args.out_dir)/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2), flush=True)
if __name__=='__main__': main()
