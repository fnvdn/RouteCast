"""Build request-aware multi-model caches without hidden states or router logits."""
from __future__ import annotations
import argparse,hashlib,json,time
from collections import Counter,defaultdict
from pathlib import Path
import torch
from routecast_rc.registry import default_specs,_route

def split(p,r):
    u=int.from_bytes(hashlib.sha1(str(p.relative_to(r)).encode()).digest()[:4],'big')/2**32
    return 'train' if u<.70 else ('val' if u<.85 else 'test')

def load_request(path,layers):
    raw=json.loads(path.read_text(encoding='utf-8')); first=raw[0]
    n=max((len(v) for v in first.values() if v),default=0); prefill=[]
    for i in range(n):
        row={}
        for l in layers:
            v=first.get(str(l)); row[l]=[int(x) for x in v[min(i,len(v)-1)]] if v else None
        prefill.append(row)
    decode=[]
    for rec in raw[1:]: decode.append({l:_route(rec.get(str(l)),'last') for l in layers})
    return prefill,decode

def entropy(p):
    q=p[p>0].float(); return float(-(q*q.log()).sum()/torch.log(torch.tensor(float(len(p))))) if len(q) else 0.

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data-root',type=Path,required=True); ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--limit-per-model',type=int,default=100); ap.add_argument('--history',type=int,default=16); a=ap.parse_args()
    a.output.mkdir(parents=True,exist_ok=True); specs=default_specs(a.data_root); manifest={'version':3,'history':a.history,'models':{},'items':[]}
    print('detected_models='+json.dumps([s.to_dict() for s in specs]),flush=True)
    for spec in specs:
        root=Path(spec.root); paths=sorted(p for p in root.rglob('*.json') if '.cache' not in p.parts)[:a.limit_per_model]
        train=[p for p in paths if split(p,root)=='train']; pairs=list(zip(spec.active_layers[:-1],spec.active_layers[1:])); pop=defaultdict(Counter)
        adj=defaultdict(lambda:defaultdict(Counter)); skip=defaultdict(lambda:defaultdict(Counter)); start=time.time()
        for fi,p in enumerate(train,1):
            pre,dec=load_request(p,spec.active_layers)
            for row in pre+dec:
                for pi,(src,dst) in enumerate(pairs):
                    cur,tgt=row.get(src),row.get(dst)
                    if cur is None or tgt is None: continue
                    pop[dst].update(tgt)
                    for e in cur: adj[(src,dst)][e].update(tgt)
                    if pi>0:
                        prev=pairs[pi-1][0]; earlier=row.get(prev)
                        if earlier is not None:
                            for e in earlier: skip[(prev,dst)][e].update(tgt)
            if fi==1 or fi%25==0 or fi==len(train): print(f'{spec.name} statistics={fi}/{len(train)} elapsed={time.time()-start:.1f}s',flush=True)
        maxl=max(spec.active_layers); pop_t=torch.zeros(maxl+1,spec.num_experts)
        for l,c in pop.items():
            z=sum(c.values()) or 1
            for e,n in c.items(): pop_t[l,e]=n/z
        def matrices(source):
            out={}
            for key,d in source.items():
                t=torch.zeros(spec.num_experts,spec.num_experts)
                for e,c in d.items():
                    z=sum(c.values()) or 1
                    for dst,n in c.items(): t[e,dst]=n/z
                out[key]=t
            return out
        adj_t,skip_t=matrices(adj),matrices(skip); model_dir=a.output/spec.name; model_dir.mkdir(parents=True,exist_ok=True)
        for fi,p in enumerate(paths,1):
            pre,dec=load_request(p,spec.active_layers); pre_freq=torch.zeros(maxl+1,spec.num_experts)
            for row in pre:
                for l,route in row.items():
                    if route is not None: pre_freq[l,route]+=1
            pre_freq/=pre_freq.sum(1,keepdim=True).clamp_min(1)
            static=[]; hist=[]; context=[]; labels=[]
            for di,row in enumerate(dec):
                timeline=pre+dec[:di]
                for pi,(src,dst) in enumerate(pairs):
                    cur,tgt=row.get(src),row.get(dst)
                    if cur is None or tgt is None: continue
                    def project(route,matrix,fallback):
                        if route is None or matrix is None: return fallback.clone()
                        hot=torch.zeros(spec.num_experts); hot[route]=1; x=hot@matrix
                        return x/x.sum() if x.sum()>0 else fallback.clone()
                    fallback=pop_t[dst].clone();
                    if fallback.sum()<=0: fallback.fill_(1/spec.num_experts)
                    a1=project(cur,adj_t.get((src,dst)),fallback)
                    earlier=row.get(pairs[pi-1][0]) if pi>0 else None
                    a2=project(earlier,skip_t.get((pairs[pi-1][0],dst)) if pi>0 else None,fallback)
                    req=pre_freq[dst].clone(); req=req if req.sum()>0 else fallback.clone()
                    routes=[r.get(dst) for r in timeline[-a.history:]]; h=torch.zeros(a.history,spec.num_experts)
                    offset=a.history-len(routes)
                    for j,route in enumerate(routes):
                        if route is not None: h[offset+j,route]=1
                    y=torch.zeros(spec.num_experts,dtype=torch.uint8); y[tgt]=1
                    static.append(torch.stack([fallback,a1,a2,req],1).half()); hist.append(h.to(torch.uint8)); labels.append(y)
                    context.append(torch.tensor([dst/maxl,spec.num_experts/256,spec.native_topk/spec.num_experts,entropy(req),di/max(len(dec),1)],dtype=torch.float16))
            name=f'{fi-1:05d}.pt'; torch.save({'static':torch.stack(static),'history':torch.stack(hist),'context':torch.stack(context),'labels':torch.stack(labels)},model_dir/name)
            manifest['items'].append({'model':spec.name,'file':str(Path(spec.name)/name),'split':split(p,root),'samples':len(labels),'source':str(p.relative_to(root))})
            if fi==1 or fi%10==0 or fi==len(paths): print(f'{spec.name} cache={fi}/{len(paths)} samples={len(labels)} elapsed={time.time()-start:.1f}s',flush=True)
        manifest['models'][spec.name]=spec.to_dict()
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8'); print(f'cache_complete={a.output}',flush=True)
if __name__=='__main__': main()
