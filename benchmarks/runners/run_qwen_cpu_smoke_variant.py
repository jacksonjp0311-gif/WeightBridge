import gc, json, os, re, sys, time
import psutil, torch
from transformers import LogitsProcessor, LogitsProcessorList, AutoTokenizer, Qwen3_5ForConditionalGeneration
model_path, label, out_path = sys.argv[1:4]
PROBLEMS = [
    {'id':'arith_42','prompt':'What is 18 + 24?','answer':'42'},
    {'id':'distance_150','prompt':'A train travels at 60 kilometers per hour for 2.5 hours. How far does it travel?','answer':'150'},
]
class PresencePenalty(LogitsProcessor):
    def __init__(self, value): self.value=value
    def __call__(self, input_ids, scores):
        for i, row in enumerate(input_ids):
            scores[i, torch.unique(row)] -= self.value
        return scores

def tail_dup8(ids):
    if len(ids) < 16: return 0.0
    split=max(0,len(ids)-128)
    tail=ids[split:]
    prev=ids[max(0,split-128):split]
    grams=lambda x: {tuple(x[i:i+8]) for i in range(max(0,len(x)-7))}
    p=grams(prev)
    t=[tuple(tail[i:i+8]) for i in range(max(0,len(tail)-7))]
    return sum(g in p for g in t)/len(t) if t else 0.0

torch.set_num_threads(16)
torch.manual_seed(0)
proc=psutil.Process()
t0=time.perf_counter()
tokenizer=AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model=Qwen3_5ForConditionalGeneration.from_pretrained(model_path, dtype=torch.bfloat16, device_map='cpu', local_files_only=True)
model.eval()
load_s=time.perf_counter()-t0
rows=[]
for problem in PROBLEMS:
    user=problem['prompt']+'\nPlease reason step by step, and put your final answer within \\boxed{}.'
    tokens=tokenizer.apply_chat_template([{'role':'user','content':user}], tokenize=True, add_generation_prompt=True, enable_thinking=True, return_dict=True, return_tensors='pt')
    input_ids=tokens['input_ids']
    tokens={k:v.to('cpu') if torch.is_tensor(v) else v for k,v in tokens.items()}
    torch.manual_seed(0)
    t1=time.perf_counter()
    with torch.inference_mode():
        generated=model.generate(**tokens, do_sample=True, temperature=0.6, top_p=0.95, top_k=20,
            max_new_tokens=256, use_cache=True, logits_processor=LogitsProcessorList([PresencePenalty(1.5)]),
            eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
    elapsed=time.perf_counter()-t1
    out_ids=generated[0,input_ids.shape[1]:].tolist()
    text=tokenizer.decode(out_ids, skip_special_tokens=True)
    matches=re.findall(r'\\boxed\s*\{([^{}]*)\}',text)
    parsed=matches[-1].strip() if matches else None
    eos=tokenizer.eos_token_id in out_ids
    rows.append({'id':problem['id'],'variant':label,'seed':0,'prompt_tokens':int(input_ids.shape[1]),
       'output_tokens':len(out_ids),'finish_reason':'stop' if eos else 'length','correct':parsed==problem['answer'],
       'boxed':parsed,'tail_duplicate_8_rate':round(tail_dup8(out_ids),6),'seconds':round(elapsed,3),
       'tokens_per_second':round(len(out_ids)/elapsed,3) if elapsed else None,'text':text})
    print(json.dumps(rows[-1],ensure_ascii=False),flush=True)
summary={'variant':label,'load_seconds':round(load_s,3),'parameter_count':sum(p.numel() for p in model.parameters()),
  'peak_process_rss_gib':round(proc.memory_info().peak_wset/1024**3,3) if hasattr(proc.memory_info(),'peak_wset') else round(proc.memory_info().rss/1024**3,3),
  'elapsed_seconds':round(time.perf_counter()-t0,3),'device':'cpu','dtype':'bfloat16','max_new_tokens':256,
  'sampling':{'temperature':0.6,'top_p':0.95,'top_k':20,'presence_penalty':1.5,'seed':0},'rows':rows}
with open(out_path,'w',encoding='utf-8') as f: json.dump(summary,f,ensure_ascii=False,indent=2)
print(json.dumps({'summary_path':out_path,'summary':{k:v for k,v in summary.items() if k!='rows'}},sort_keys=True),flush=True)
