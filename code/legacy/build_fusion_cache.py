"""Build reusable per-request fusion features with batched CUDA GRU inference."""
from __future__ import annotations
import argparse, hashlib, json, time
from collections import Counter, defaultdict
from pathlib import Path
import torch
from torch import nn

def bucket(p,r): return int.from_bytes(hashlib.sha1(str(p.relative_to(r)).encode()).digest()[:4],'big')/2**32
def split(p,r):
    b=bucket(p,r); return 'train' if b<.70 else ('val' if b<.85 else 'test')

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,required=True); ap.add_argument('--model',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--limit',type=int,default=100); ap.add_argument('--history',type=int,default=8); ap.add_argument('--batch-size',type=int,default=4096); ap.add_argument('--stats-cache',type=Path,default=None,help='reuse train statistics across history values'); a=ap.parse_args()
    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); print(f'device={dev}',flush=True)
    paths=sorted(p for p in a.root.rglob('*.json') if '.cache' not in p.parts)[:a.limit]; train=[p for p in paths if split(p,a.root)=='train']; print(f'files={len(paths)} train={len(train)}',flush=True)
    pop=defaultdict(Counter); trans=defaultdict(lambda:defaultdict(Counter))
    if a.stats_cache is not None and a.stats_cache.exists():
        cached=torch.load(a.stats_cache,map_location='cpu'); pop=defaultdict(Counter,{int(l):Counter(v) for l,v in cached['pop'].items()}); trans=defaultdict(lambda:defaultdict(Counter),{int(l):defaultdict(Counter,{int(e):Counter(v) for e,v in d.items()}) for l,d in cached['trans'].items()}); print(f'statistics_reused={a.stats_cache}',flush=True)
    else:
        for i,p in enumerate(train,1):
            with p.open(encoding='utf-8') as f: tr=json.load(f)
            for tok in tr:
                for l in range(93):
                    dst=tok[str(l+1)][0]; pop[l+1].update(dst)
                    for e in tok[str(l)][0]: trans[l][e].update(dst)
            print(f'statistics={i}/{len(train)}',flush=True)
        if a.stats_cache is not None:
            a.stats_cache.parent.mkdir(parents=True,exist_ok=True); torch.save({'pop':{str(l):dict(c) for l,c in pop.items()},'trans':{str(l):{str(e):dict(c) for e,c in d.items()} for l,d in trans.items()}},a.stats_cache); print(f'statistics_saved={a.stats_cache}',flush=True)
    pop_tensor=torch.zeros(94,128)
    for l,c in pop.items():
        z=sum(c.values()) or 1
        for e,n in c.items(): pop_tensor[l,e]=n/z
    # Vectorized transition tensor: [source_layer, current_expert, next_expert].
    # This replaces the former Python loop over every sample and expert.
    trans_tensor=torch.zeros(93,128,128)
    for l,d in trans.items():
        for e,c in d.items():
            for dst,n in c.items(): trans_tensor[l,e,dst]=n
    ck=torch.load(a.model,map_location='cpu'); gru=nn.GRU(222,128,batch_first=True).to(dev); head=nn.Linear(128,128).to(dev); gru.load_state_dict(ck['gru']); head.load_state_dict(ck['head']); gru.eval(); head.eval(); a.output.mkdir(parents=True,exist_ok=True)
    manifest=[]; start=time.time()
    for fi,p in enumerate(paths,1):
        with p.open(encoding='utf-8') as f: tr=json.load(f)
        seqs=[]; layers=[]; currents=[]; targets=[]
        for t in range(a.history-1,len(tr)):
            for l in range(93):
                seqs.append([tr[t-a.history+1+j][str(l)][0] for j in range(a.history)]); layers.append(l); currents.append(tr[t][str(l)][0]); targets.append(tr[t][str(l+1)][0])
        chunks=[]; labels=[]
        for st in range(0,len(seqs),a.batch_size):
            sr=torch.tensor(seqs[st:st+a.batch_size],dtype=torch.long); ls=layers[st:st+a.batch_size]; cs=currents[st:st+a.batch_size]; ts=targets[st:st+a.batch_size]; b=len(sr); x=torch.zeros(b,a.history,222)
            bi=torch.arange(b)[:,None,None].expand(-1,a.history,8); hi=torch.arange(a.history)[None,:,None].expand(b,-1,8); x[bi,hi,sr]=1; lid=torch.tensor(ls); x[torch.arange(b)[:,None],torch.arange(a.history)[None,:],128+lid[:,None]]=1
            with torch.inference_mode(), torch.autocast(device_type=dev.type,dtype=torch.float16,enabled=dev.type=='cuda'): g=torch.sigmoid(head(gru(x.to(dev))[0][:,-1])).float().cpu()
            ps=pop_tensor[lid+1].to(dev); cur_hot=torch.zeros(b,128,device=dev)
            for r,cur in enumerate(cs): cur_hot[r,cur]=1
            tm=trans_tensor[torch.tensor(ls)].to(dev)
            trs=torch.bmm(cur_hot.unsqueeze(1),tm).squeeze(1)
            trs=trs/(trs.sum(1,keepdim=True)+1e-8)
            y=torch.zeros(b,128,dtype=torch.uint8)
            for r,tgt in enumerate(ts): y[r,tgt]=1
            chunks.append(torch.stack([ps.cpu(),trs.cpu(),g],2).half()); labels.append(y)
            print(f'cache_file={fi}/{len(paths)} batch={st//a.batch_size+1} samples={min(st+a.batch_size,len(seqs))}/{len(seqs)} elapsed={time.time()-start:.1f}s',flush=True)
        name=f'{fi-1:05d}.pt'; torch.save({'features':torch.cat(chunks),'labels':torch.cat(labels)},a.output/name); manifest.append({'file':name,'split':split(p,a.root),'source':str(p.relative_to(a.root)),'samples':len(seqs)})
        print(f'cache_file_complete={fi}/{len(paths)} saved={name}',flush=True)
    (a.output/'manifest.json').write_text(json.dumps({'history':a.history,'limit':a.limit,'items':manifest},ensure_ascii=False,indent=2),encoding='utf-8'); print(f'cache_complete={a.output}',flush=True)
if __name__=='__main__': main()
