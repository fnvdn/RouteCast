"""Train one RouteCast model jointly across heterogeneous MoE traces."""
from __future__ import annotations
import argparse, json, random, time
from collections import defaultdict
from pathlib import Path
import torch
from torch import nn

from routecast_rc.universal_model import UniversalRouteCast


def rank_loss(scores, labels, hard_negatives=16):
    pos=labels.bool(); n=min(hard_negatives,scores.shape[1]-1)
    neg=scores.masked_fill(pos,-1e9).topk(n,1).values
    max_pos=max(1,int(labels.sum(1).max().item()))
    ps=scores.masked_fill(~pos,-1e9).topk(max_pos,1).values
    valid=(ps>-1e8).float()
    raw=torch.nn.functional.softplus(-(ps.unsqueeze(2)-neg.unsqueeze(1)))
    return (raw*valid.unsqueeze(2)).sum()/(valid.sum()*n+1e-8)


def objective(scores, labels, bce_weight, rank_weight, cost_weight):
    labels=labels.float(); k=labels.sum(1).mean().clamp_min(1)
    pos_weight=((scores.shape[1]-k)/k).detach()
    bce=nn.functional.binary_cross_entropy_with_logits(scores,labels,pos_weight=pos_weight)
    target=labels/labels.sum(1,keepdim=True).clamp_min(1)
    listwise=-(target*scores.log_softmax(1)).sum(1).mean()
    ranking=rank_loss(scores,labels)
    waste=(scores.sigmoid()*(1-labels)).sum(1)/scores.sigmoid().sum(1).clamp_min(1e-6)
    return bce_weight*(.5*bce+.5*listwise)+rank_weight*ranking+cost_weight*waste.mean()


def batches(data,batch_size,device):
    n=len(data['labels'])
    order=torch.randperm(n) if n else torch.empty(0,dtype=torch.long)
    for st in range(0,n,batch_size):
        ix=order[st:st+batch_size]
        yield (data['static'][ix].float().to(device,non_blocking=True),
               data['history'][ix].float().to(device,non_blocking=True),
               data['context'][ix].float().to(device,non_blocking=True),
               data['labels'][ix].float().to(device,non_blocking=True))


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--cache',type=Path,required=True)
    ap.add_argument('--epochs',type=int,default=3); ap.add_argument('--batch-size',type=int,default=2048)
    ap.add_argument('--lr',type=float,default=8e-4); ap.add_argument('--bce-weight',type=float,default=.55)
    ap.add_argument('--rank-weight',type=float,default=.45); ap.add_argument('--cost-weight',type=float,default=.0)
    ap.add_argument('--save',type=Path,required=True); ap.add_argument('--seed',type=int,default=2026)
    a=ap.parse_args(); random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); m=json.loads((a.cache/'manifest.json').read_text(encoding='utf-8'))
    items=m['items']; train=[x for x in items if x['split']=='train']; totals=defaultdict(int)
    for x in train: totals[x['model']]+=x['samples']
    total=sum(totals.values()); balance={name:total/(len(totals)*n) for name,n in totals.items()}
    model=UniversalRouteCast(m['history']).to(dev); opt=torch.optim.AdamW(model.parameters(),lr=a.lr)
    print(f'device={dev} models={list(m["models"])} train_files={len(train)} balance={balance}',flush=True)
    for ep in range(a.epochs):
        model.train(); random.shuffle(train); running=seen=0; start=time.time()
        for fi,item in enumerate(train,1):
            d=torch.load(a.cache/item['file'],map_location='cpu')
            for static,history,context,y in batches(d,a.batch_size,dev):
                with torch.autocast(device_type=dev.type,dtype=torch.float16,enabled=dev.type=='cuda'):
                    scores,_=model(static,history,context)
                    loss=balance[item['model']]*objective(scores,y,a.bce_weight,a.rank_weight,a.cost_weight)
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); running+=loss.item()*len(y); seen+=len(y)
            if fi==1 or fi%5==0 or fi==len(train):
                mem=torch.cuda.max_memory_allocated()/1e6 if dev.type=='cuda' else 0
                print(f'epoch={ep+1}/{a.epochs} file={fi}/{len(train)} model={item["model"]} samples={seen} loss={running/max(seen,1):.6f} gpu_mb={mem:.1f} elapsed={time.time()-start:.1f}s',flush=True)
    def evaluate(split):
        result={}; model.eval(); start=time.time()
        for name,spec in m['models'].items():
            subset=[x for x in items if x['split']==split and x['model']==name]; hits=defaultdict(float); predicted=defaultdict(float); actual=0; examples=0
            ks=sorted(set([1,2,4,8,12,16,spec['native_topk']]))
            with torch.inference_mode():
                for fi,item in enumerate(subset,1):
                    d=torch.load(a.cache/item['file'],map_location='cpu')
                    n=len(d['labels'])
                    for st in range(0,n,a.batch_size):
                        static=d['static'][st:st+a.batch_size].float().to(dev); history=d['history'][st:st+a.batch_size].float().to(dev)
                        context=d['context'][st:st+a.batch_size].float().to(dev); y=d['labels'][st:st+a.batch_size].to(dev)
                        scores,_=model(static,history,context); actual+=y.sum().item(); examples+=len(y)
                        for k in ks:
                            kk=min(k,scores.shape[1]); idx=scores.topk(kk,1).indices; hits[k]+=y.gather(1,idx).sum().item(); predicted[k]+=len(y)*kk
                    if fi==1 or fi%10==0 or fi==len(subset): print(f'{split} model={name} file={fi}/{len(subset)} elapsed={time.time()-start:.1f}s',flush=True)
            result[name]={'examples':examples,'native_topk':spec['native_topk'],'metrics':{str(k):{'recall':hits[k]/max(actual,1),'precision':hits[k]/max(predicted[k],1)} for k in ks}}
        return result
    val=evaluate('val'); test=evaluate('test'); report={'seed':a.seed,'history':m['history'],'val':val,'test':test}
    a.save.parent.mkdir(parents=True,exist_ok=True); torch.save({'model':model.state_dict(),'history':m['history'],'report':report},a.save)
    a.save.with_suffix('.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('\n================ CROSS-MODEL SUMMARY ================',flush=True)
    for name,spec in m['models'].items():
        k=str(spec['native_topk']); vm=val[name]['metrics'][k]; tm=test[name]['metrics'][k]
        print(f'{name:18s} native_K={k:>2s} val_recall={vm["recall"]:.6f} test_recall={tm["recall"]:.6f} test_precision={tm["precision"]:.6f}',flush=True)
    print(f'model_saved={a.save}',flush=True); print(f'report_saved={a.save.with_suffix(".json")}',flush=True)
    print('=====================================================',flush=True)
if __name__=='__main__': main()
