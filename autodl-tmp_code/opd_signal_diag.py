import os, sys, json, time, math
from pathlib import Path
import torch
import torch.nn.functional as F

CODE = Path(os.environ.get('CODE_DIR', '/root/RWKV-LM/RWKV7-math500_trl_doc_g1f1p5b_align_traintempv3_opd_masked_20260611'))
sys.path.insert(0, str(CODE))
os.chdir(str(CODE))
os.environ['RWKV_HEAD_SIZE_A']='64'
os.environ['RWKV_MY_TESTING']='x070'
os.environ['RWKV_TRAIN_TYPE']='fullstate'
os.environ['RWKV_CTXLEN']='8192'
os.environ['FUSED_KERNEL']='0'
os.environ['WKV']='cuda'

from main import load_train_model_rwkv7_cuda
from reference.utils import TRIE_TOKENIZER
from utils import read_jsonl, build_prompt, set_seed
from reward import calculate_reward_details
from train import GRPOConfig
from infer import AlbatrossBatchInference
from stateful_rollout import StatefulTrainRollout

set_seed(42)
device='cuda'
model_path='/root/autodl-tmp/rwkv_models/ms_g1_1p5b/rwkv7-g1f-1.5b-20260419-ctx8192.pth'
teacher_path='/root/autodl-tmp/rwkv_models/ms_g1_7p2b/rwkv7-g1f-7.2b-20260414-ctx8192.pth'
tok_path='/root/RWKV-LM/rwkv_vocab_v20230424.txt'
data_path='/root/Albatross/faster3a_2605/dataset/MATH500.jsonl'
out_dir=Path('/root/autodl-tmp/logs/math500_opd_signal_diag_'+time.strftime('%Y%m%d_%H%M%S'))
out_dir.mkdir(parents=True, exist_ok=True)
Path('/tmp/math500_opd_signal_diag_latest').write_text(str(out_dir))

tok=TRIE_TOKENIZER(tok_path)
encode=lambda s: tok.encode(s)
def decode(ids):
    try: return tok.decode(ids, utf8_errors='replace')
    except TypeError: return tok.decode(ids)

print('loading student', flush=True)
student,_=load_train_model_rwkv7_cuda(model_path, device=device, ctx_len=8192, train_type='fullstate', load_dtype='bf16')
student.eval()
print('loading teacher', flush=True)
teacher,_=load_train_model_rwkv7_cuda(teacher_path, device=device, ctx_len=8192, train_type='fullstate', load_dtype='bf16')
teacher.eval()
for p in teacher.parameters(): p.requires_grad=False

cfg=GRPOConfig(num_questions=4,samples_per_question=8,max_new_tokens=1024,temperature=0.8,top_p=1.0,top_k=0,rollout_forward_batch=8,tune_mode='full',reward_mode='trl_doc',prompt_mode='trl_doc',min_tokens=1,max_tokens=1024,length_weight=0.0,zstd_penalty_weight=0.0,ngram_penalty=0.0)
infer=AlbatrossBatchInference(StatefulTrainRollout(student, device), student, encode, decode, device, cfg)

data=read_jsonl(data_path)[:4]
prompts=[build_prompt(x['problem'], mode='trl_doc') for x in data]
prompt_tokens=[encode(p) for p in prompts]
answers=[x['answer'] for x in data]
print('generating', flush=True)
with torch.no_grad():
    comp_tokens, old_logps, texts, trunc = infer.generate_group_parallel(prompt_tokens, group_size=8, max_new_tokens=1024, temperature=0.8, top_p=1.0, top_k=0, stop_on_user=True, stop_on_boxed=False, stop_on_repeat_ngram=False, presence_penalty=0.0, frequency_penalty=0.0, alpha_decay=1.0)

rows=[]
print('scoring', flush=True)
for qi in range(len(data)):
  for si in range(8):
    idx=qi*8+si
    comp=comp_tokens[idx]
    txt=texts[idx]
    reward, ok, fmt, details = calculate_reward_details(txt, answers[qi], len(comp), min_tokens=1, max_tokens=1024, length_weight=0.0, repeat_ngram=False, repeat_penalty=0.0, zstd_penalty_weight=0.0, reward_mode='trl_doc')
    if not comp:
        continue
    seq=prompt_tokens[qi]+comp
    x=torch.tensor(seq[:-1], dtype=torch.long, device=device).unsqueeze(0)
    tgt=torch.tensor(seq[1:], dtype=torch.long, device=device).unsqueeze(0)
    pl=len(prompt_tokens[qi]); cl=len(comp); sidx=pl-1; eidx=sidx+cl
    with torch.no_grad():
        sh=student.forward_hidden(x); th=teacher.forward_hidden(x)
        slog=student.project_logits(sh[:,sidx:eidx,:]).squeeze(0).float()
        tlog=teacher.project_logits(th[:,sidx:eidx,:]).squeeze(0).float()
        ttgt=tgt[:,sidx:eidx].squeeze(0)
        slogp=F.log_softmax(slog, dim=-1)
        tlogp=F.log_softmax(tlog, dim=-1)
        topv, topi=torch.topk(tlog, k=64, dim=-1)
        tprob=F.softmax(tlog, dim=-1)
        topmass=tprob.gather(-1, topi).sum(dim=-1)
        in_top=(topi == ttgt[:,None]).any(dim=-1).float()
        student_act=slogp.gather(-1, ttgt[:,None]).squeeze(-1)
        teacher_act=tlogp.gather(-1, ttgt[:,None]).squeeze(-1)
        slogp_top=slogp.gather(-1, topi)
        tprob_top=F.softmax(topv, dim=-1)
        kl_top=F.kl_div(slogp_top, tprob_top, reduction='none').sum(dim=-1)
    row={
      'question': qi,
      'sample': si,
      'correct': bool(ok),
      'reward': float(reward),
      'length': len(comp),
      'truncated': bool(trunc[idx]),
      'teacher_action_logp_mean': float(teacher_act.mean().item()),
      'student_action_logp_mean': float(student_act.mean().item()),
      'teacher_minus_student_action_logp': float((teacher_act-student_act).mean().item()),
      'action_in_teacher_top64_rate': float(in_top.mean().item()),
      'teacher_top64_mass_mean': float(topmass.mean().item()),
      'current_top64_kl_mean': float(kl_top.mean().item()),
      'text_head': txt[:300],
      'extracted': details.get('extracted_answer'),
    }
    rows.append(row)
    with (out_dir/'rows.jsonl').open('a', encoding='utf-8') as f:
        f.write(json.dumps(row, ensure_ascii=False)+'\n')
    del x,tgt,sh,th,slog,tlog,slogp,tlogp,topv,topi,tprob,topmass,in_top,student_act,teacher_act,slogp_top,tprob_top,kl_top
    torch.cuda.empty_cache()

def avg(sub, key):
    vals=[r[key] for r in sub if isinstance(r.get(key), (int,float)) and math.isfinite(r[key])]
    return sum(vals)/len(vals) if vals else None
correct=[r for r in rows if r['correct']]
wrong=[r for r in rows if not r['correct']]
summary={'out_dir': str(out_dir),'n': len(rows),'correct_n': len(correct),'wrong_n': len(wrong),'correct_rate': len(correct)/max(1,len(rows))}
for name,sub in [('correct',correct),('wrong',wrong),('all',rows)]:
    summary[name]={
      'len_mean': avg(sub,'length'),
      'teacher_action_logp_mean': avg(sub,'teacher_action_logp_mean'),
      'student_action_logp_mean': avg(sub,'student_action_logp_mean'),
      'teacher_minus_student_action_logp': avg(sub,'teacher_minus_student_action_logp'),
      'action_in_teacher_top64_rate': avg(sub,'action_in_teacher_top64_rate'),
      'teacher_top64_mass_mean': avg(sub,'teacher_top64_mass_mean'),
      'current_top64_kl_mean': avg(sub,'current_top64_kl_mean'),
    }
(out_dir/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
print('OPD_SIGNAL_DIAG '+json.dumps(summary, ensure_ascii=False), flush=True)

