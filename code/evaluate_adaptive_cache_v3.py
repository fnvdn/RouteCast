"""Adaptive mass budgeting and cache-aware prefetching for frozen RouteCast V3.

No hardware latency/bandwidth values are assumed.  Validation chooses the
smallest average dynamic K that preserves a requested fraction of fixed-native
Recall. Cache-aware replay then prefetches only when a candidate's
temperature-scaled ranking weight exceeds the value of the entry that would
be evicted. These unit-sum weights control score concentration; they are not
interpreted as calibrated categorical probabilities for multi-label routers.
"""
from __future__ import annotations
import argparse,csv,heapq,json,time
from collections import defaultdict
from pathlib import Path
import torch
from routecast_rc.universal_model_v3 import UniversalRouteCastV3
from calibrate_routecast_v3 import batches,unpack_binary

def ranking_weights(logits,temp):
    # Unit-sum weights make score concentration comparable and support a
    # cumulative-prefix rule. For Top-K>1 routing they are not categorical
    # probabilities. Temperature leaves the expert ranking unchanged.
    return torch.softmax(logits.float()/temp,dim=1)

def infer(net,d,batch,device,temp):
    out=[]
    with torch.inference_mode():
        for static,hist,ctx,_ in batches(d,batch,device):
            with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=='cuda'): z,_=net(static,hist,ctx)
            out.append(ranking_weights(z,temp).cpu())
    return torch.cat(out)

def labels_of(d):
    y=d['labels']; width=int(d.get('_num_experts',d['static'].shape[1]))
    return unpack_binary(y,width).float() if d.get('_binary_packed',False) else y.float()

def dynamic_choice(p,mass,max_k):
    v,idx=p.sort(1,descending=True); cap=min(max_k,p.shape[1]); cdf=v[:,:cap].cumsum(1)
    # mass=1 is an explicit safe fallback to the complete allowed budget.
    k=torch.full((len(p),),cap,dtype=torch.long,device=p.device) if mass>=1.0-1e-12 else ((cdf<mass).sum(1)+1).clamp(max=cap)
    return idx[:,:cap],v[:,:cap],k

def select_abstention(items,root,model,net,batch,device,temp,mass,max_k,thresholds,waste_penalty):
    totals={q:[0.,0.,0.] for q in thresholds}; subset=[x for x in items if x['model']==model and x['split']=='val']; start=time.time()
    for fi,it in enumerate(subset,1):
        d=torch.load(root/it['file'],map_location='cpu'); y=labels_of(d); p=infer(net,d,batch,device,temp); idx,v,k=dynamic_choice(p,mass,max_k); truth=y.gather(1,idx); rank=torch.arange(idx.shape[1])[None,:]; allowed=rank<k[:,None]
        for q in thresholds:
            issued=allowed&(v>=q); hit=(truth*issued).sum().item(); pred=issued.sum().item(); totals[q][0]+=hit; totals[q][1]+=pred; totals[q][2]+=len(y)
        if fi==1 or fi%10==0 or fi==len(subset): print(f'{model} val_abstention={fi}/{len(subset)} elapsed={time.time()-start:.1f}s',flush=True)
    curve={}
    for q,(hit,pred,n) in totals.items():
        waste=pred-hit; curve[q]={'issued_per_example':pred/n,'precision':hit/max(pred,1),'hits_per_example':hit/n,'waste_per_example':waste/n,'utility_per_example':(hit-waste_penalty*waste)/n}
    # A threshold above one is included, so abstaining everywhere is always a
    # valid zero-utility fallback when all speculative prefetching is harmful.
    best=max(thresholds,key=lambda q:(curve[q]['utility_per_example'],-curve[q]['issued_per_example'],q))
    return best,curve

def quality(items,root,model,net,batch,device,temp,masses,max_k,native,split,progress=True):
    sums={m:[0.,0.,0.,0.] for m in masses}; fixed=[0.,0.,0.]; subset=[x for x in items if x['model']==model and x['split']==split]; start=time.time()
    for fi,it in enumerate(subset,1):
        d=torch.load(root/it['file'],map_location='cpu'); y=labels_of(d); p=infer(net,d,batch,device,temp); actual=y.sum().item(); fixed_idx=p.topk(min(native,p.shape[1]),1).indices
        fixed[0]+=y.gather(1,fixed_idx).sum().item(); fixed[1]+=actual; fixed[2]+=len(y)*native
        for mass in masses:
            idx,_,k=dynamic_choice(p,mass,max_k); mask=torch.arange(idx.shape[1])[None,:]<k[:,None]; hit=(y.gather(1,idx)*mask).sum().item(); sums[mass][0]+=hit; sums[mass][1]+=actual; sums[mass][2]+=k.sum().item(); sums[mass][3]+=len(y)
        if progress and (fi==1 or fi%10==0 or fi==len(subset)): print(f'{model} {split}_quality={fi}/{len(subset)} elapsed={time.time()-start:.1f}s',flush=True)
    curve={m:{'avg_k':x[2]/x[3],'recall':x[0]/max(x[1],1),'precision':x[0]/max(x[2],1)} for m,x in sums.items()}
    return {'avg_k':native,'recall':fixed[0]/max(fixed[1],1),'precision':fixed[0]/max(fixed[2],1)},curve

class Cache:
    def __init__(self,capacity):
        self.capacity=capacity; self.entries={}; self.heap=[]; self.clock=0; self.version=0; self.prefetched=set(); self.hit=self.miss=self.prefetch=self.useful=self.waste=0; self.deadline_rejected=0
    def _set(self,key,value):
        self.clock+=1; self.version+=1; state=(value,self.clock,self.version); self.entries[key]=state; heapq.heappush(self.heap,(value,self.clock,self.version,key))
    def victim(self):
        while self.heap:
            value,last,version,key=self.heap[0]
            if key in self.entries and self.entries[key]==(value,last,version): return key
            heapq.heappop(self.heap)
        raise RuntimeError('cache heap is empty')
    def insert(self,key,value,is_prefetch,protect=False,margin=0.0):
        if key in self.entries: self._set(key,max(value,self.entries[key][0])); return True
        if len(self.entries)>=self.capacity:
            old=self.victim()
            if protect and value<=self.entries[old][0]+margin: return False
            if old in self.prefetched: self.prefetched.remove(old); self.waste+=1
            del self.entries[old]
        self._set(key,value)
        if is_prefetch: self.prefetched.add(key); self.prefetch+=1
        return True
    def access(self,key,value):
        if key in self.entries:
            self.hit+=1; self._set(key,max(value,self.entries[key][0]))
            if key in self.prefetched: self.prefetched.remove(key); self.useful+=1
        else: self.miss+=1; self.insert(key,value,False)
    def refresh(self,score_by_key,decay):
        for key,(old,_,_) in list(self.entries.items()): self._set(key,score_by_key.get(key,old*decay))
    def finish(self): self.waste+=len(self.prefetched); self.prefetched.clear()
    def metrics(self):
        demand=self.hit+self.miss
        return {'hit_rate':self.hit/max(demand,1),'on_demand_loads':self.miss,'prefetch_loads':self.prefetch,'useful_prefetches':self.useful,'wasted_prefetches':self.waste,'prefetch_precision':self.useful/max(self.prefetch,1),'total_transfers':self.miss+self.prefetch,'demand_accesses':demand,'deadline_rejected':self.deadline_rejected}

def token_groups(d):
    coord=d['context'][:,4].float(); starts=torch.cat((torch.tensor([True]),coord[1:]!=coord[:-1])); gid=starts.cumsum(0)-1
    return [(gid==t).nonzero().flatten() for t in range(int(gid[-1])+1)]

def replay(d,p,mass,native,max_k,capacity_mult,policy,decay,margin,
           abstain_threshold,transfer_budget=None):
    """Replay one request.

    With transfer_budget=None, every admitted candidate is available before
    demand. This is the idealized cache-event replay used by the original
    paper. An integer transfer_budget limits successful speculative loads per
    token window and provides a deadline-aware sensitivity interface.
    """
    y=labels_of(d); hist=d['history']; width=p.shape[1]
    if d.get('_binary_packed',False): hist=unpack_binary(hist,width)
    current=hist[:,-1]; max_layer=max(1,int(round(float(d['context'][:,0].max())*10000)))
    groups=token_groups(d); layers=(d['context'][:,0].float()*max_layer).round().long(); capacity=max(1,len(torch.unique(layers))*native*capacity_mult); c=Cache(capacity)
    # Warm with the observed current route for the first predicted token.
    for row in groups[0].tolist():
        l=int(layers[row]);
        for e in current[row].nonzero().flatten().tolist(): c.insert((l,e),0.,False)
    for rows in groups:
        row_by_layer={int(layers[row]):row for row in rows.tolist()}
        # Refresh only resident entries; materializing L x E Python objects at
        # every token dominates replay time on 128/256-expert models.
        score_by_key={(l,e):float(p[row_by_layer[l],e]) for (l,e) in c.entries if l in row_by_layer}
        if policy!='LRU': c.refresh(score_by_key,decay)
        if policy!='LRU':
            if policy=='fixed_native': idx=p[rows].topk(min(native,width),1).indices; vals=p[rows].gather(1,idx); ks=torch.full((len(rows),),idx.shape[1],dtype=torch.long)
            else: idx,vals,ks=dynamic_choice(p[rows],mass,max_k)
            candidates=[]
            for j,row in enumerate(rows.tolist()):
                l=int(layers[row])
                for q in range(int(ks[j])): candidates.append((float(vals[j,q]),(l,int(idx[j,q]))))
            completed_this_token=0
            for value,key in sorted(candidates,reverse=True):
                if policy=='adaptive_cache' and value<abstain_threshold: continue
                if key in c.entries: continue
                if transfer_budget is not None and completed_this_token>=transfer_budget:
                    c.deadline_rejected+=1
                    continue
                inserted=c.insert(key,value,True,protect=(policy=='adaptive_cache'),margin=margin)
                if inserted: completed_this_token+=1
        for row in rows.tolist():
            l=int(layers[row])
            for e in y[row].nonzero().flatten().tolist(): c.access((l,e),0.0 if policy=='LRU' else float(p[row,e]))
    c.finish(); return c.metrics()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--cache',type=Path,required=True); ap.add_argument('--model',type=Path,required=True); ap.add_argument('--calibration',type=Path,required=True); ap.add_argument('--save',type=Path,required=True)
    ap.add_argument('--masses',type=float,nargs='+',default=[i/100 for i in range(1,101)])
    ap.add_argument('--selection-mode',choices=['budget','quality'],default='budget',help='budget: maximize Recall under average-K cap; quality: minimize average K under Recall-retention target')
    ap.add_argument('--budget-multiplier',type=float,default=1.0,help='Average-K cap as a multiple of native K in budget mode')
    ap.add_argument('--recall-retention',type=float,default=.98); ap.add_argument('--max-k',type=int,default=32); ap.add_argument('--capacity-multipliers',type=int,nargs='+',default=[1,2,4]); ap.add_argument('--cache-decay',type=float,default=.90); ap.add_argument('--eviction-margin',type=float,default=0.0)
    ap.add_argument('--abstain-thresholds',type=float,nargs='+',default=[0,.01,.02,.05,.10,.15,.20,.30,.40,.50,.60,.70,.80,.90,1.01]); ap.add_argument('--waste-penalty',type=float,default=1.0,help='Unit cost of a wrong speculative prefetch relative to a useful prefetch')
    ap.add_argument('--deadline-transfer-budgets',type=int,nargs='*',default=[],
                    help='Optional completed expert transfers allowed per token window; adds deadline-aware adaptive-cache sensitivity policies')
    ap.add_argument('--batch-size',type=int,default=1024); a=ap.parse_args()
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); man=json.loads((a.cache/'manifest.json').read_text(encoding='utf-8')); items=man['items']; cal=json.loads(a.calibration.read_text(encoding='utf-8')); ck=torch.load(a.model,map_location='cpu'); net=UniversalRouteCastV3(ck['history'],disabled_branches=ck.get('disabled_branches',[])).to(device); net.load_state_dict(ck['model']); net.eval()
    report={'method':'marginally calibrated predictor plus validation-selected score-mass budget and cache-aware prefetch','selection_quantity':'temperature-scaled unit-sum ranking weight (not categorical probability for multi-label routing)','selection_mode':a.selection_mode,'recall_retention':a.recall_retention,'budget_multiplier':a.budget_multiplier,'models':{}}
    print(f'device={device} predictor_frozen=True selection_mode={a.selection_mode} recall_retention={a.recall_retention} budget_multiplier={a.budget_multiplier}',flush=True)
    for model,spec in man['models'].items():
        native=spec['native_topk']; temp=float(cal['temperatures'][model]); policy_max_k=min(a.max_k,max(1,int(native*a.budget_multiplier))) if a.selection_mode=='budget' else a.max_k
        fixed_val,curve=quality(items,a.cache,model,net,a.batch_size,device,temp,a.masses,policy_max_k,native,'val'); target=fixed_val['recall']*a.recall_retention; budget_cap=native*a.budget_multiplier
        if a.selection_mode=='budget':
            feasible=[m for m in a.masses if curve[m]['avg_k']<=budget_cap+1e-9 and curve[m]['recall']>=target]
            # With a hard per-example K cap, retain the requested Recall using
            # the smallest average K. If infeasible, take the best Recall that
            # respects the cap.
            chosen=min(feasible,key=lambda m:(curve[m]['avg_k'],-curve[m]['recall'],m)) if feasible else max(a.masses,key=lambda m:(curve[m]['recall'],-curve[m]['avg_k'],-m))
        else:
            feasible=[m for m in a.masses if curve[m]['recall']>=target]
            chosen=min(feasible,key=lambda m:(curve[m]['avg_k'],-curve[m]['recall'],m)) if feasible else max(a.masses,key=lambda m:curve[m]['recall'])
        abstain_threshold,abstain_curve=select_abstention(items,a.cache,model,net,a.batch_size,device,temp,chosen,policy_max_k,a.abstain_thresholds,a.waste_penalty)
        fixed_test,test_curve=quality(items,a.cache,model,net,a.batch_size,device,temp,[chosen],policy_max_k,native,'test'); system=defaultdict(lambda:defaultdict(lambda:defaultdict(float))); subset=[x for x in items if x['model']==model and x['split']=='test']; start=time.time()
        deadline_budgets=sorted(set(a.deadline_transfer_budgets))
        if any(x<0 for x in deadline_budgets): raise ValueError('deadline transfer budgets must be non-negative')
        for fi,it in enumerate(subset,1):
            d=torch.load(a.cache/it['file'],map_location='cpu'); p=infer(net,d,a.batch_size,device,temp)
            for mult in a.capacity_multipliers:
                for policy in ('LRU','fixed_native','adaptive_mass','adaptive_cache'):
                    out=replay(d,p,chosen,native,policy_max_k,mult,policy,a.cache_decay,a.eviction_margin,abstain_threshold)
                    for k,v in out.items(): system[str(mult)][policy][k]+=v
                for budget in deadline_budgets:
                    label=f'adaptive_deadline_b{budget}'
                    out=replay(d,p,chosen,native,policy_max_k,mult,'adaptive_cache',a.cache_decay,a.eviction_margin,abstain_threshold,transfer_budget=budget)
                    for k,v in out.items(): system[str(mult)][label][k]+=v
            if fi==1 or fi%10==0 or fi==len(subset): print(f'{model} cache_replay={fi}/{len(subset)} elapsed={time.time()-start:.1f}s',flush=True)
        sysout={}
        for mult,policies in system.items():
            sysout[mult]={}
            for policy,x in policies.items():
                demand=x['demand_accesses']; pref=x['prefetch_loads']; x['hit_rate']=(demand-x['on_demand_loads'])/max(demand,1); x['prefetch_precision']=x['useful_prefetches']/max(pref,1); sysout[mult][policy]=dict(x)
        report['models'][model]={'temperature':temp,'native_topk':native,'policy_max_k':policy_max_k,'replay_semantics':'idealized cache-event replay unless policy name contains adaptive_deadline','deadline_transfer_budgets':deadline_budgets,'val_fixed':fixed_val,'target_recall':target,'budget_cap':budget_cap,'selected_mass':chosen,'selected_abstain_threshold':abstain_threshold,'waste_penalty':a.waste_penalty,'val_abstention_curve':{str(k):v for k,v in abstain_curve.items()},'val_curve':{str(k):v for k,v in curve.items()},'test_fixed':fixed_test,'test_adaptive':test_curve[chosen],'system':sysout}
        print(f'\n{model} mode={a.selection_mode} selected_mass={chosen:.2f} abstain_threshold={abstain_threshold:.2f} budget_cap={budget_cap:.4f} target={target:.6f} val_recall={curve[chosen]["recall"]:.6f} val_avg_k={curve[chosen]["avg_k"]:.4f} test_recall={test_curve[chosen]["recall"]:.6f} test_avg_k={test_curve[chosen]["avg_k"]:.4f}',flush=True)
        for mult,policies in sysout.items():
            for policy,x in policies.items(): print(f'  cache={mult}x {policy:14s} hit={x["hit_rate"]:.6f} transfers={int(x["total_transfers"])} prefetch_precision={x["prefetch_precision"]:.6f}',flush=True)
    a.save.parent.mkdir(parents=True,exist_ok=True); a.save.write_text(json.dumps(report,indent=2),encoding='utf-8'); csv_path=a.save.with_suffix('.csv')
    with csv_path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f); w.writerow(['model','selection_mode','budget_cap','selected_mass','abstain_threshold','test_avg_k','test_recall','test_precision','capacity','policy','hit_rate','total_transfers','prefetch_precision','wasted_prefetches'])
        for model,x in report['models'].items():
            for mult,policies in x['system'].items():
                for policy,m in policies.items(): w.writerow([model,a.selection_mode,x['budget_cap'],x['selected_mass'],x['selected_abstain_threshold'],x['test_adaptive']['avg_k'],x['test_adaptive']['recall'],x['test_adaptive']['precision'],mult,policy,m['hit_rate'],m['total_transfers'],m['prefetch_precision'],m['wasted_prefetches']])
    print(f'\nreport_saved={a.save}\nsummary_saved={csv_path}',flush=True)
if __name__=='__main__': main()
