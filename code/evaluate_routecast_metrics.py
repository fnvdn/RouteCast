"""Unified RouteCast evaluation with request-level Recall/Precision/MRR/NDCG."""
from __future__ import annotations
import argparse,csv,json,math,time
from collections import defaultdict
from pathlib import Path
import torch
from routecast_rc.universal_model_v3 import UniversalRouteCastV3
from routecast_rc.tcn_baseline import HistoryTCN
from train_universal_routecast_v3 import BUDGETS,amp_dtype,assert_finite,iter_batches

def fresh(): return defaultdict(lambda:{'hits':0.0,'truth':0.0,'predicted':0.0,'examples':0,'rr':0.0,'ndcg':0.0})

def update(acc,scores,y,budgets):
    truth=y.bool(); maxk=min(max(budgets),scores.shape[1]); order=scores.topk(maxk,1).indices
    ranked_truth=truth.gather(1,order); truth_count=truth.sum(1)
    # Full-list first-positive rank without materializing a complete argsort.
    best_true=scores.masked_fill(~truth,float('-inf')).amax(1,keepdim=True)
    first_rank=1+(scores>best_true).sum(1); rr=torch.where(truth_count>0,first_rank.float().reciprocal(),torch.zeros_like(first_rank,dtype=torch.float))
    for k in budgets:
        kk=min(k,scores.shape[1]); rel=ranked_truth[:,:kk].float(); hits=rel.sum()
        discounts=1/torch.log2(torch.arange(2,kk+2,device=scores.device,dtype=torch.float32))
        dcg=(rel*discounts).sum(1); ideal_n=torch.minimum(truth_count,torch.tensor(kk,device=scores.device))
        prefix=torch.cat([torch.zeros(1,device=scores.device),discounts.cumsum(0)])
        idcg=prefix[ideal_n]; ndcg=torch.where(idcg>0,dcg/idcg,torch.zeros_like(dcg))
        row=acc[k]; row['hits']+=float(hits); row['truth']+=float(truth_count.sum()); row['predicted']+=len(y)*kk
        row['examples']+=len(y); row['rr']+=float(rr.sum()); row['ndcg']+=float(ndcg.sum())

def finish(row):
    return {'examples':row['examples'],'hits':row['hits'],'truth':row['truth'],'predicted':row['predicted'],
            'recall':row['hits']/max(row['truth'],1),'precision':row['hits']/max(row['predicted'],1),
            'avg_hits':row['hits']/max(row['examples'],1),'mrr':row['rr']/max(row['examples'],1),'ndcg':row['ndcg']/max(row['examples'],1)}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--cache',type=Path,required=True); p.add_argument('--model',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True); p.add_argument('--split',choices=('val','test'),default='test')
    p.add_argument('--batch-size',type=int,default=256); p.add_argument('--progress-every',type=int,default=10); a=p.parse_args()
    manifest=json.loads((a.cache/'manifest.json').read_text(encoding='utf-8')); ck=torch.load(a.model,map_location='cpu')
    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if ck.get('model_type')=='history_tcn':
        channels=int(ck.get('report',{}).get('channels',24))
        net=HistoryTCN(ck.get('history',manifest['history']),channels=channels).to(dev)
        method='TCN-only'
    else:
        net=UniversalRouteCastV3(ck.get('history',manifest['history']),disabled_branches=ck.get('disabled_branches',())).to(dev)
        method='RouteCast V3'
    net.load_state_dict(ck['model']); net.eval(); report={'method':'RouteCast V3','checkpoint':str(a.model),'split':a.split,'models':{}}; request_rows=[]
    report['method']=method
    started=time.time()
    with torch.inference_mode():
        for name,spec in manifest['models'].items():
            items=[x for x in manifest['items'] if x['model']==name and x['split']==a.split]; total=fresh()
            for fi,item in enumerate(items,1):
                data=torch.load(a.cache/item['file'],map_location='cpu'); local=fresh()
                for static,hist,ctx,y in iter_batches(data,a.batch_size,dev,False):
                    with torch.autocast(device_type=dev.type,dtype=amp_dtype(dev),enabled=dev.type=='cuda'): scores,_=net(static,hist,ctx)
                    assert_finite(scores,'scores',model=name,file=fi,split=a.split); update(total,scores,y,BUDGETS[name]); update(local,scores,y,BUDGETS[name])
                request_key=str(item.get('source',item['file'])).replace('\\','/')
                for k in BUDGETS[name]: request_rows.append({'model':name,'request_file':request_key,'k':k,**finish(local[k])})
                if fi==1 or fi%a.progress_every==0 or fi==len(items): print(f'metrics_progress model={name} file={fi}/{len(items)} elapsed={time.time()-started:.1f}s',flush=True)
            report['models'][name]={'native_topk':spec['native_topk'],'budgets':{str(k):finish(total[k]) for k in BUDGETS[name]}}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    csvp=a.output.with_name(a.output.stem+'_request_level.csv')
    with csvp.open('w',newline='',encoding='utf-8-sig') as f: w=csv.DictWriter(f,fieldnames=list(request_rows[0])); w.writeheader(); w.writerows(request_rows)
    print('\n================ UNIFIED METRICS ================',flush=True)
    for name,x in report['models'].items():
        k=str(x['native_topk']); z=x['budgets'][k]; print(f'{name:18s} K={int(k):2d} recall={z["recall"]:.6f} precision={z["precision"]:.6f} mrr={z["mrr"]:.6f} ndcg={z["ndcg"]:.6f}',flush=True)
    print(f'json_saved={a.output}\nrequest_metrics_saved={csvp}',flush=True)

if __name__=='__main__': main()
