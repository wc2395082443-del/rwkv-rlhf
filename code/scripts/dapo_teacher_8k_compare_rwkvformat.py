import argparse,json,os,re,time,statistics
from pathlib import Path
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
from math_verify import parse,verify

def repeat_flag(text):
    words=text.split()
    for n in (30,20,15,12,10,8):
        for i in range(max(0,len(words)-3*n)):
            if words[i:i+n]==words[i+n:i+2*n]==words[i+2*n:i+3*n]: return True
    return False

def is_correct(answer,text):
    try:
        gold=parse(f'$\\boxed{{{answer}}}$'); pred=parse(text)
        return bool(pred and verify(gold,pred,strict=False))
    except Exception: return False

ap=argparse.ArgumentParser(); ap.add_argument('--model'); ap.add_argument('--name'); ap.add_argument('--data'); ap.add_argument('--out'); ap.add_argument('--limit',type=int,default=256); ap.add_argument('--batch',type=int,default=8)
a=ap.parse_args(); out=Path(a.out)/a.name; out.mkdir(parents=True,exist_ok=True)
rows=[]
with open(a.data,encoding='utf-8') as f:
    for line in f:
        if line.strip(): rows.append(json.loads(line))
rows=rows[:a.limit]
tok=AutoTokenizer.from_pretrained(a.model,trust_remote_code=True)
tok.padding_side='left'
if tok.pad_token_id is None: tok.pad_token_id=tok.eos_token_id
model=AutoModelForCausalLM.from_pretrained(a.model,dtype=torch.bfloat16,trust_remote_code=True,attn_implementation='sdpa').to('cuda').eval()
result_path=out/'generations.jsonl'; done=set()
if result_path.exists():
    for line in result_path.open(encoding='utf-8'):
        try: done.add(json.loads(line)['unique_id'])
        except: pass
started=time.time(); total_new=0
for start in range(0,len(rows),a.batch):
    batch=[r for r in rows[start:start+a.batch] if r.get('unique_id',str(start)) not in done]
    if not batch: continue
    prompts=[]
    for r in batch:
        user=r['problem'].strip()+'\nPlease reason step by step. Close the reasoning with </think>, then output exactly <answer>final answer</answer> and immediately end the response.'
        prompts.append('User: '+user+'\nAssistant: <think>\n')
    enc=tok(prompts,return_tensors='pt',padding=True,truncation=True,max_length=512).to('cuda')
    prompt_len=enc.input_ids.shape[1]; max_new=max(1,8192-prompt_len)
    t0=time.time()
    with torch.inference_mode():
        seq=model.generate(**enc,max_new_tokens=max_new,do_sample=True,temperature=0.6,top_p=0.95,use_cache=True,pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id)
    elapsed=time.time()-t0
    outputs=[]
    for i,r in enumerate(batch):
        new_ids=seq[i,prompt_len:]; text=tok.decode(new_ids,skip_special_tokens=True)
        n=int((new_ids!=tok.pad_token_id).sum().item()); total_new+=n
        item={'unique_id':r.get('unique_id'),'problem':r['problem'],'answer':r['answer'],'completion':text,'generated_tokens':n,'truncated':n>=max_new-1,'correct':is_correct(r['answer'],text),'repeat':repeat_flag(text),'elapsed_batch_sec':elapsed}
        outputs.append(item)
    with result_path.open('a',encoding='utf-8') as f:
        for item in outputs: f.write(json.dumps(item,ensure_ascii=False)+'\n')
    all_done=len(done)+min(start+a.batch,len(rows)); wall=time.time()-started
    print(f'{a.name} done={all_done}/{len(rows)} batch_s={elapsed:.1f} total_tps={total_new/max(wall,1):.1f}',flush=True)
allrows=[json.loads(x) for x in result_path.open(encoding='utf-8')]
summary={'model':a.model,'name':a.name,'n':len(allrows),'accuracy':sum(x['correct'] for x in allrows)/len(allrows),'trunc_rate':sum(x['truncated'] for x in allrows)/len(allrows),'repeat_rate':sum(x['repeat'] for x in allrows)/len(allrows),'avg_generated_tokens':statistics.mean(x['generated_tokens'] for x in allrows),'elapsed_sec':time.time()-started,'token_per_sec':total_new/max(time.time()-started,1),'ctx_limit':8192,'temperature':0.6,'top_p':0.95}
json.dump(summary,(out/'summary.json').open('w'),ensure_ascii=False,indent=2); print(json.dumps(summary,ensure_ascii=False),flush=True)