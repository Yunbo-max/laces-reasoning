import json, sys, torch
from pathlib import Path
sys.path.insert(0, str(Path('/inspire/hdd/global_user/zhangjiaquan-253108540222/research/laces-reasoning')))
sys.path.insert(0, str(Path('/inspire/hdd/global_user/zhangjiaquan-253108540222/research/laces-reasoning/scripts/eval')))
import importlib.util
spec=importlib.util.spec_from_file_location('rocb','/inspire/hdd/global_user/zhangjiaquan-253108540222/research/laces-reasoning/scripts/eval/run_open_context_bench.py')
R=importlib.util.module_from_spec(spec); spec.loader.exec_module(R)
from transformers import AutoModelForCausalLM, AutoTokenizer
MD='/inspire/hdd/global_user/zhangjiaquan-253108540222/backup/laces-reasoning/models/RWKV7-Goose-World3-2.9B-HF'
tok=AutoTokenizer.from_pretrained(MD, trust_remote_code=True, local_files_only=True)
model=AutoModelForCausalLM.from_pretrained(MD, trust_remote_code=True, torch_dtype=torch.bfloat16, local_files_only=True).to('cuda').eval()
D='/inspire/hdd/global_user/zhangjiaquan-253108540222/hf_cache/hub/datasets--xiaowu0162--longmemeval/snapshots/2ec2a557f339b6c0369619b1ed5793734cc87533/longmemeval_oracle'
items=json.load(open(D))[:50]
@torch.no_grad()
def loop(prefix, n=128):
    out=model(input_ids=prefix, use_cache=True, return_dict=True)
    cache, logits = out.past_key_values, out.logits[0,-1]
    ids=[]
    for i in range(n):
        t=int(logits.argmax())
        if t==getattr(tok,'eos_token_id',None): break
        ids.append(t)
        if i+1<n:
            out=model(input_ids=torch.tensor([[t]],device=prefix.device,dtype=torch.long), past_key_values=cache, use_cache=True, return_dict=True)
            cache, logits = out.past_key_values, out.logits[0,-1]
    return ids
tot=0.0; deg=0; res=[]
for k,it in enumerate(items):
    prompt=R.build_memeval_prompt(it)
    ids=tok(prompt, return_tensors='pt', add_special_tokens=False).input_ids.to('cuda')
    g=loop(ids)
    pred=tok.decode(g, skip_special_tokens=True).strip()
    s=R.longmemeval_score_approx(pred, str(it['answer']), it.get('question',''))
    tot+=s
    lines=[l.strip() for l in pred.split('\n') if l.strip()]
    d=bool(lines and len(set(lines))<=2 and len(lines)>=5)
    deg+=d
    res.append({'gold':str(it['answer']),'pred':pred,'score':s,'degenerate':d})
    print(f'[loop] {k+1}/50 score={s:.3f} running={tot/(k+1):.4f}', flush=True)
json.dump(res, open('results/benchmarks/longctx/rwkv7-world3__memeval_loopdecode.json','w'), ensure_ascii=False, indent=2)
print(f'FINAL loop score={tot/50:.4f}  degenerate={deg}/50')
