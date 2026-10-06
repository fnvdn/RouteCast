"""Paired request-level bootstrap for RouteCast versus a baseline."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np

NATIVE={'qwen3':8,'deepseek_r1':8,'llama4_maverick':1}

def read(path,task=None):
    with path.open(encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
    out={}
    for r in rows:
        # Paper-Heatmap exports contain a task column; neural evaluators do not.
        # Apply the task filter only when that schema is present.
        if task is not None and 'task' in r and r.get('task')!=task: continue
        model=r['model']; k=int(r['k'])
        if k!=NATIVE[model]: continue
        out[(model,r['request_file'].replace('\\','/'))]=r
    return out

def pooled(rows,metric,index):
    if metric=='recall': return rows[index,0].sum(1)/np.maximum(rows[index,1].sum(1),1)
    if metric=='precision': return rows[index,0].sum(1)/np.maximum(rows[index,2].sum(1),1)
    col=3 if metric=='mrr' else 4
    return (rows[index,col]*rows[index,5]).sum(1)/np.maximum(rows[index,5].sum(1),1)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--routecast',type=Path,required=True); p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--baseline-task',default='cross_token'); p.add_argument('--repeats',type=int,default=10000); p.add_argument('--seed',type=int,default=2026); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    ours=read(a.routecast); base=read(a.baseline,a.baseline_task); rng=np.random.default_rng(a.seed); report={'seed':a.seed,'repeats':a.repeats,'sampling_unit':'request','models':{}}
    for model in NATIVE:
        keys=sorted(set(k for k in ours if k[0]==model)&set(k for k in base if k[0]==model)); n=len(keys)
        if not n: raise ValueError(f'no paired requests for {model}')
        def matrix(source):
            values=[]
            for key in keys:
                r=source[key]; values.append([float(r['hits']),float(r['truth']),float(r['predicted']),float(r['mrr']),float(r['ndcg']),float(r['examples'])])
            return np.asarray(values,dtype=np.float64)
        x=matrix(ours); y=matrix(base); idx=rng.integers(0,n,size=(a.repeats,n)); model_report={'requests':n,'native_k':NATIVE[model],'metrics':{}}
        for metric in ('recall','precision','mrr','ndcg'):
            delta=pooled(x,metric,idx)-pooled(y,metric,idx); point=float(pooled(x,metric,np.arange(n)[None,:])[0]-pooled(y,metric,np.arange(n)[None,:])[0])
            model_report['metrics'][metric]={'delta':point,'ci95':[float(np.quantile(delta,.025)),float(np.quantile(delta,.975))],'probability_positive':float((delta>0).mean())}
        report['models'][model]=model_report
        z=model_report['metrics']['recall']; print(f'{model:18s} requests={n} recall_delta={z["delta"]:+.6f} ci95=[{z["ci95"][0]:+.6f},{z["ci95"][1]:+.6f}] p_positive={z["probability_positive"]:.4f}',flush=True)
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(report,indent=2),encoding='utf-8'); print(f'bootstrap_saved={a.output}',flush=True)

if __name__=='__main__': main()
