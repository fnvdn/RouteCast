"""Trace-level reproduction of MoE-Infinity EAM prediction and cache replay.

The implementation follows the paper's request-level Expert Activation Matrix
Collection (EAMC): completed training-request EAMs are retained, a partial
online request EAM is cosine-matched against the collection, and the nearest
historical EAMs are aggregated into future expert likelihoods.  The predictor
is evaluated on the same cross-token targets as RouteCast.  A predictor-neutral
cache replay then compares LRU, EAM and RouteCast under identical capacities.
"""
from __future__ import annotations
import argparse,csv,json,time
from collections import defaultdict
from pathlib import Path
import torch
from routecast_rc.universal_model_v3 import UniversalRouteCastV3

BUDGETS={'qwen3':[8,12,16],'deepseek_r1':[8,12,16],'llama4_maverick':[1,2,4,8]}

def unpack_binary(x,width):
    shifts=torch.arange(8,dtype=torch.uint8)
    return (((x.unsqueeze(-1)>>shifts)&1).reshape(*x.shape[:-1],-1))[...,:width]

def load_arrays(path,active_layers):
    d=torch.load(path,map_location='cpu'); e=int(d.get('_num_experts',d['static'].shape[1]))
    h=d['history']; y=d['labels']
    if d.get('_binary_packed',False): h=unpack_binary(h,e); y=unpack_binary(y,e)
    current=h[:,-1].float(); labels=y.float()
    # Some traces omit a layer for an individual token.  Recover token groups
    # and layer IDs from the coordinates stored by the cache builder instead
    # of assuming that every token has a rectangular L x E route matrix.
    token_coord=d['context'][:,4].float(); layer_coord=d['context'][:,0].float()
    starts=torch.cat((torch.tensor([True]),token_coord[1:]!=token_coord[:-1]))
    token_id=starts.cumsum(0)-1; tokens=int(token_id[-1])+1
    max_layer=max(active_layers); layer_to_slot={x:i for i,x in enumerate(active_layers)}
    layer_id=(layer_coord*max_layer).round().long()
    dense_cur=torch.zeros(tokens,len(active_layers),e); dense_y=torch.zeros_like(dense_cur)
    valid=torch.zeros(tokens,len(active_layers),dtype=torch.bool); flat_index=torch.full((tokens,len(active_layers)),-1,dtype=torch.long)
    for row,(t,l) in enumerate(zip(token_id.tolist(),layer_id.tolist())):
        if l not in layer_to_slot: continue
        slot=layer_to_slot[l]; dense_cur[t,slot]=current[row]; dense_y[t,slot]=labels[row]; valid[t,slot]=True; flat_index[t,slot]=row
    return d,dense_cur,dense_y,valid,flat_index

def full_eam(current,labels):
    return current.sum(0)+labels[-1]

def build_collection(items,root,model,active_layers,device):
    refs=[]; subset=[x for x in items if x['model']==model and x['split']=='train']; start=time.time()
    for i,it in enumerate(subset,1):
        _,cur,y,_,_=load_arrays(root/it['file'],active_layers); refs.append(full_eam(cur,y))
        if i==1 or i%25==0 or i==len(subset): print(f'{model} eam_collection={i}/{len(subset)} elapsed={time.time()-start:.1f}s',flush=True)
    raw=torch.stack(refs).to(device); flat=raw.flatten(1); norm=flat/flat.norm(dim=1,keepdim=True).clamp_min(1e-9)
    return raw,norm

def eam_scores(current,reference_raw,reference_norm,neighbors):
    partial=torch.zeros_like(reference_raw[0]); outputs=[]
    for token in range(current.shape[0]):
        partial+=current[token].to(partial.device)
        query=partial.flatten(); query=query/query.norm().clamp_min(1e-9)
        sim=reference_norm@query; k=min(neighbors,len(sim)); index=sim.topk(k).indices
        predicted=reference_raw[index].mean(0)
        predicted=predicted/predicted.sum(1,keepdim=True).clamp_min(1e-9)
        outputs.append(predicted.cpu())
    return torch.stack(outputs)

def recall_for(scores,labels,k,valid=None):
    index=scores.topk(min(k,scores.shape[-1]),-1).indices
    hits=labels.gather(-1,index).sum(-1)
    if valid is not None: hits=hits[valid]; labels=labels[valid]
    return hits.sum().item(),labels.sum().item(),hits.numel()*min(k,scores.shape[-1])

def tune_neighbors(items,root,model,active_layers,native,candidates,raw,norm):
    totals={k:[0.,0.] for k in candidates}; subset=[x for x in items if x['model']==model and x['split']=='val']; start=time.time()
    for i,it in enumerate(subset,1):
        _,cur,y,valid,_=load_arrays(root/it['file'],active_layers)
        for k in candidates:
            score=eam_scores(cur,raw,norm,k); hit,actual,_=recall_for(score,y,native,valid); totals[k][0]+=hit; totals[k][1]+=actual
        if i==1 or i%10==0 or i==len(subset): print(f'{model} eam_val={i}/{len(subset)} elapsed={time.time()-start:.1f}s',flush=True)
    recalls={k:h/max(a,1) for k,(h,a) in totals.items()}; best=max(candidates,key=lambda k:(recalls[k],-k))
    return best,recalls

def load_routecast(checkpoint,device):
    ckpt=torch.load(checkpoint,map_location='cpu'); disabled=ckpt.get('disabled_branches',[])
    net=UniversalRouteCastV3(ckpt['history'],disabled_branches=disabled).to(device); net.load_state_dict(ckpt['model']); net.eval(); return net

def routecast_scores(data,net,device,batch):
    n=len(data['labels']); outputs=[]; width=int(data.get('_num_experts',data['static'].shape[1]))
    with torch.inference_mode():
        for st in range(0,n,batch):
            static=data['static'][st:st+batch].float(); hist=data['history'][st:st+batch]; context=data['context'][st:st+batch].float()
            if data.get('_binary_packed',False): hist=unpack_binary(hist,width)
            with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=='cuda'):
                score,_=net(static.to(device),hist.float().to(device),context.to(device))
            outputs.append(score.float().cpu())
    return torch.cat(outputs)

class ReplayCache:
    def __init__(self,capacity):
        self.capacity=capacity; self.entries={}; self.clock=0; self.prefetched=set(); self.prefetch_loads=0; self.useful=0; self.wasted=0; self.hits=0; self.misses=0
    def _evict(self):
        if len(self.entries)<self.capacity:return
        victim=min(self.entries,key=lambda x:(self.entries[x][0],self.entries[x][1]))
        if victim in self.prefetched: self.prefetched.remove(victim); self.wasted+=1
        del self.entries[victim]
    def insert(self,key,priority,prefetch=False):
        self.clock+=1
        if key in self.entries: self.entries[key]=(priority,self.clock); return
        self._evict(); self.entries[key]=(priority,self.clock)
        if prefetch: self.prefetched.add(key); self.prefetch_loads+=1
    def access(self,key,priority):
        self.clock+=1
        if key in self.entries:
            self.hits+=1; self.entries[key]=(priority,self.clock)
            if key in self.prefetched: self.prefetched.remove(key); self.useful+=1
        else:
            self.misses+=1; self.insert(key,priority,False)
    def finish(self): self.wasted+=len(self.prefetched); self.prefetched.clear()
    def metrics(self):
        demand=self.hits+self.misses
        return {'hit_rate':self.hits/max(demand,1),'on_demand_loads':self.misses,'prefetch_loads':self.prefetch_loads,'useful_prefetches':self.useful,'wasted_prefetches':self.wasted,'prefetch_precision':self.useful/max(self.prefetch_loads,1),'total_transfers':self.misses+self.prefetch_loads,'demand_accesses':demand}

def replay(scores,labels,current,valid,native,capacity_multiplier,predictive):
    tokens,layers,experts=labels.shape; capacity=min(layers*experts,max(1,layers*native*capacity_multiplier)); cache=ReplayCache(capacity)
    # Warm-up state represents experts used by the last prefill/current token.
    for l in range(layers):
        for e in current[0,l].nonzero().flatten().tolist(): cache.insert((l,e),0.,False)
    for t in range(tokens):
        if predictive:
            flat=[]
            for l in range(layers):
                if valid[t,l]:
                    for e in scores[t,l].topk(min(native,experts)).indices.tolist(): flat.append((float(scores[t,l,e]),l,e))
            for priority,l,e in sorted(flat): cache.insert((l,e),priority,True)
        for l in range(layers):
            if not valid[t,l]: continue
            for e in labels[t,l].nonzero().flatten().tolist():
                priority=float(scores[t,l,e]) if predictive else 0.; cache.access((l,e),priority)
    cache.finish(); out=cache.metrics(); out['capacity_entries']=capacity; out['capacity_multiplier']=capacity_multiplier; return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--cache',type=Path,required=True); ap.add_argument('--routecast-model',type=Path,required=True); ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--neighbors',type=int,nargs='+',default=[1,3,5,10]); ap.add_argument('--capacity-multipliers',type=int,nargs='+',default=[1,2,4]); ap.add_argument('--batch-size',type=int,default=512); a=ap.parse_args()
    manifest=json.loads((a.cache/'manifest.json').read_text(encoding='utf-8')); items=manifest['items']; device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); net=load_routecast(a.routecast_model,device)
    report={'method':'MoE-Infinity EAMC trace reproduction','neighbors_candidates':a.neighbors,'models':{}}
    print(f'device={device}',flush=True)
    for model,spec in manifest['models'].items():
        active_layers=spec['active_layers']; layers=len(active_layers); native=spec['native_topk']; raw,norm=build_collection(items,a.cache,model,active_layers,device); best,val=tune_neighbors(items,a.cache,model,active_layers,native,a.neighbors,raw,norm)
        metrics={k:[0.,0.,0.] for k in BUDGETS[model]}; system=defaultdict(lambda:defaultdict(lambda:defaultdict(float))); subset=[x for x in items if x['model']==model and x['split']=='test']; start=time.time()
        for fi,it in enumerate(subset,1):
            data,cur,y,valid,flat_index=load_arrays(a.cache/it['file'],spec['active_layers']); eam=eam_scores(cur,raw,norm,best)
            rc_flat=routecast_scores(data,net,device,a.batch_size); rc=torch.zeros_like(y); rc[valid]=rc_flat[flat_index[valid]]
            for k in BUDGETS[model]:
                hit,actual,predicted=recall_for(eam,y,k,valid); metrics[k][0]+=hit; metrics[k][1]+=actual; metrics[k][2]+=predicted
            for mult in a.capacity_multipliers:
                for method,scores,predictive in [('LRU',torch.zeros_like(y),False),('MoE-Infinity EAM',eam,True),('RouteCast',rc,True)]:
                    out=replay(scores,y,cur,valid,native,mult,predictive)
                    for key,value in out.items(): system[str(mult)][method][key]+=value
            if fi==1 or fi%10==0 or fi==len(subset): print(f'{model} test={fi}/{len(subset)} elapsed={time.time()-start:.1f}s',flush=True)
        pred={str(k):{'recall':h/max(actual,1),'precision':h/max(predicted,1)} for k,(h,actual,predicted) in metrics.items()}
        sysout={}
        for mult,methods in system.items():
            sysout[mult]={}
            for method,x in methods.items():
                demand=x['demand_accesses']; prefetch=x['prefetch_loads']
                x['hit_rate']=x['demand_accesses'] and (demand-x['on_demand_loads'])/demand
                x['prefetch_precision']=x['useful_prefetches']/max(prefetch,1); sysout[mult][method]=dict(x)
        report['models'][model]={'native_topk':native,'eamc_size':len(raw),'best_neighbors':best,'val_native_recall':val[best],'prediction':pred,'system':sysout}
        print(f'{model} best_neighbors={best} val_recall={val[best]:.6f}',flush=True)
        for k,x in pred.items(): print(f'{model:18s} EAM K={int(k):2d} recall={x["recall"]:.6f} precision={x["precision"]:.6f}',flush=True)
        for mult,methods in sysout.items():
            for method,x in methods.items(): print(f'{model:18s} cache={mult}x method={method:16s} hit={x["hit_rate"]:.6f} transfers={int(x["total_transfers"])} waste={int(x["wasted_prefetches"])}',flush=True)
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    pred_csv=a.output.with_suffix('.prediction.csv'); sys_csv=a.output.with_suffix('.system.csv'); md=a.output.with_suffix('.md')
    with pred_csv.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f); w.writerow(['model','method','K','recall','precision','best_neighbors','val_native_recall'])
        for model,x in report['models'].items():
            for k,m in x['prediction'].items(): w.writerow([model,'MoE-Infinity EAM',k,m['recall'],m['precision'],x['best_neighbors'],x['val_native_recall']])
    with sys_csv.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f); w.writerow(['model','capacity_multiplier','method','hit_rate','on_demand_loads','prefetch_loads','useful_prefetches','wasted_prefetches','prefetch_precision','total_transfers'])
        for model,x in report['models'].items():
            for mult,methods in x['system'].items():
                for method,m in methods.items(): w.writerow([model,mult,method,m['hit_rate'],m['on_demand_loads'],m['prefetch_loads'],m['useful_prefetches'],m['wasted_prefetches'],m['prefetch_precision'],m['total_transfers']])
    lines=['# MoE-Infinity EAM trace-level reproduction','','## Prediction quality','','| Model | K | Recall | Precision | Best neighbors |','|---|---:|---:|---:|---:|']
    for model,x in report['models'].items():
        for k,m in x['prediction'].items(): lines.append(f'| {model} | {k} | {m["recall"]:.6f} | {m["precision"]:.6f} | {x["best_neighbors"]} |')
    lines += ['','## Cache replay','','| Model | Capacity | Method | Hit rate | Total transfers | Prefetch precision | Wasted prefetches |','|---|---:|---|---:|---:|---:|---:|']
    for model,x in report['models'].items():
        for mult,methods in x['system'].items():
            for method,m in methods.items(): lines.append(f'| {model} | {mult}x | {method} | {m["hit_rate"]:.6f} | {int(m["total_transfers"])} | {m["prefetch_precision"]:.6f} | {int(m["wasted_prefetches"])} |')
    lines += ['','> Scope: trace-level reproduction under an identical expert-entry cache model. This is not an end-to-end latency or throughput reproduction of the original offloading system.','']
    md.write_text('\n'.join(lines),encoding='utf-8')
    print(f'report_saved={a.output}\nprediction_csv_saved={pred_csv}\nsystem_csv_saved={sys_csv}\nsummary_saved={md}',flush=True)
if __name__=='__main__': main()
