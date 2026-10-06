"""Build portable route-only caches for Qwen, DeepSeek and Llama traces."""
from __future__ import annotations
import argparse, hashlib, json, time
from collections import Counter, defaultdict
from pathlib import Path
import torch

from routecast_rc.registry import default_specs, _route


def split(path, root):
    u = int.from_bytes(hashlib.sha1(str(path.relative_to(root)).encode()).digest()[:4], "big") / 2**32
    return "train" if u < .70 else ("val" if u < .85 else "test")


def records(path):
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [{int(k): _route(v, "last") for k, v in row.items()} for row in raw]


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data-root',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True); ap.add_argument('--limit-per-model',type=int,default=1000)
    ap.add_argument('--history',type=int,default=8); ap.add_argument('--chunk-size',type=int,default=65536)
    a=ap.parse_args(); a.output.mkdir(parents=True,exist_ok=True)
    specs=default_specs(a.data_root); manifest={'history':a.history,'models':{},'items':[]}
    print('detected_models='+json.dumps([s.to_dict() for s in specs]),flush=True)
    for spec in specs:
        root=Path(spec.root); paths=sorted(p for p in root.rglob('*.json') if '.cache' not in p.parts)[:a.limit_per_model]
        train=[p for p in paths if split(p,root)=='train']; pairs=list(zip(spec.active_layers[:-1],spec.active_layers[1:]))
        pop=defaultdict(Counter); trans=defaultdict(lambda:defaultdict(Counter)); start=time.time()
        for i,p in enumerate(train,1):
            for row in records(p):
                for src,dst in pairs:
                    cur,tgt=row.get(src),row.get(dst)
                    if cur is None or tgt is None: continue
                    pop[dst].update(tgt)
                    for e in cur: trans[(src,dst)][e].update(tgt)
            if i==1 or i%25==0 or i==len(train): print(f'{spec.name} statistics={i}/{len(train)} elapsed={time.time()-start:.1f}s',flush=True)
        pop_t=torch.zeros(max(spec.active_layers)+1,spec.num_experts)
        trans_t={}
        for layer,c in pop.items():
            z=sum(c.values()) or 1
            for e,n in c.items(): pop_t[layer,e]=n/z
        for pair in pairs:
            d=trans.get(pair,{})
            t=torch.zeros(spec.num_experts,spec.num_experts)
            for e,c in d.items():
                z=sum(c.values()) or 1
                for dst,n in c.items(): t[e,dst]=n/z
            trans_t[pair]=t
        model_dir=a.output/spec.name; model_dir.mkdir(parents=True,exist_ok=True); item_id=0
        for fi,p in enumerate(paths,1):
            rr=records(p); statics=[]; histories=[]; contexts=[]; labels=[]
            # Current source-layer route is known.  Temporal evidence must use
            # only PREVIOUS tokens at the destination layer; including the
            # current destination route would leak the prediction target.
            for ti in range(a.history,len(rr)):
                for src,dst in pairs:
                    cur,tgt=rr[ti].get(src),rr[ti].get(dst)
                    hist=[rr[j].get(dst) for j in range(ti-a.history,ti)]
                    if cur is None or tgt is None or any(x is None for x in hist): continue
                    cur_hot=torch.zeros(spec.num_experts); cur_hot[cur]=1
                    tr=(cur_hot @ trans_t[(src,dst)]).clamp_min(0)
                    tr=tr/tr.sum() if tr.sum()>0 else pop_t[dst].clone()
                    if tr.sum()<=0: tr.fill_(1/spec.num_experts)
                    h=torch.zeros(a.history,spec.num_experts)
                    for j,route in enumerate(hist): h[j,route]=1
                    y=torch.zeros(spec.num_experts,dtype=torch.uint8); y[tgt]=1
                    statics.append(torch.stack([pop_t[dst],tr],1).half()); histories.append(h.to(torch.uint8)); labels.append(y)
                    contexts.append(torch.tensor([dst/max(spec.active_layers),spec.num_experts/256,spec.native_topk/spec.num_experts],dtype=torch.float16))
            name=f'{item_id:05d}.pt'; torch.save({'static':torch.stack(statics) if statics else torch.empty(0,spec.num_experts,2,dtype=torch.float16),'history':torch.stack(histories) if histories else torch.empty(0,a.history,spec.num_experts,dtype=torch.uint8),'context':torch.stack(contexts) if contexts else torch.empty(0,3,dtype=torch.float16),'labels':torch.stack(labels) if labels else torch.empty(0,spec.num_experts,dtype=torch.uint8)},model_dir/name)
            manifest['items'].append({'model':spec.name,'file':str(Path(spec.name)/name),'split':split(p,root),'samples':len(labels)})
            item_id+=1
            if fi==1 or fi%10==0 or fi==len(paths): print(f'{spec.name} cache={fi}/{len(paths)} samples={len(labels)} elapsed={time.time()-start:.1f}s',flush=True)
        manifest['models'][spec.name]=spec.to_dict()
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    print(f'cache_complete={a.output}',flush=True)
if __name__=='__main__': main()
