"""Evaluate an existing universal RouteCast checkpoint at multiple budgets."""
from __future__ import annotations
import argparse, csv, json, time
from pathlib import Path
import torch

from routecast_rc.universal_model import UniversalRouteCast


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--cache',type=Path,required=True)
    ap.add_argument('--model',type=Path,required=True); ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--batch-size',type=int,default=256); ap.add_argument('--split',default='test')
    ap.add_argument('--qwen-ks',default='8,12,16'); ap.add_argument('--deepseek-ks',default='8,12,16')
    ap.add_argument('--llama-ks',default='1,2,4,8'); a=ap.parse_args()
    manifest=json.loads((a.cache/'manifest.json').read_text(encoding='utf-8'))
    ck=torch.load(a.model,map_location='cpu'); dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net=UniversalRouteCast(ck.get('history',manifest['history'])).to(dev); net.load_state_dict(ck['model']); net.eval()
    budgets={'qwen3':[int(x) for x in a.qwen_ks.split(',')],
             'deepseek_r1':[int(x) for x in a.deepseek_ks.split(',')],
             'llama4_maverick':[int(x) for x in a.llama_ks.split(',')]}
    rows=[]; report={'model':str(a.model),'cache':str(a.cache),'split':a.split,'results':{}}
    print(f'device={dev} split={a.split}',flush=True)
    for name,ks in budgets.items():
        if dev.type=='cuda': torch.cuda.empty_cache()
        subset=[x for x in manifest['items'] if x['model']==name and x['split']==a.split]
        hit={k:0.0 for k in ks}; predicted={k:0 for k in ks}; actual=examples=0; start=time.time()
        with torch.inference_mode():
            for fi,item in enumerate(subset,1):
                d=torch.load(a.cache/item['file'],map_location='cpu'); n=len(d['labels'])
                for st in range(0,n,a.batch_size):
                    static=d['static'][st:st+a.batch_size].float().to(dev); history=d['history'][st:st+a.batch_size].float().to(dev)
                    context=d['context'][st:st+a.batch_size].float().to(dev); y=d['labels'][st:st+a.batch_size].to(dev)
                    with torch.autocast(device_type=dev.type,dtype=torch.float16,enabled=dev.type=='cuda'):
                        scores,_=net(static,history,context)
                    actual+=y.sum().item(); examples+=len(y)
                    for k in ks:
                        kk=min(k,scores.shape[1]); idx=scores.topk(kk,1).indices
                        hit[k]+=y.gather(1,idx).sum().item(); predicted[k]+=len(y)*kk
                print(f'budget_progress model={name} file={fi}/{len(subset)} elapsed={time.time()-start:.1f}s',flush=True)
        native=manifest['models'][name]['native_topk']; model_rows=[]
        for k in ks:
            recall=hit[k]/max(actual,1); precision=hit[k]/max(predicted[k],1)
            avg_hits=hit[k]/max(examples,1); wasted=1-precision
            row={'model':name,'native_topk':native,'prefetch_k':k,'examples':examples,'recall':recall,'precision':precision,'avg_hits':avg_hits,'wasted_prefetch_ratio':wasted}
            rows.append(row); model_rows.append(row)
        report['results'][name]=model_rows
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.with_suffix('.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    with a.output.with_suffix('.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    print('\n================ BUDGET CURVE SUMMARY ================',flush=True)
    for row in rows:
        print(f"{row['model']:18s} K={row['prefetch_k']:2d} recall={row['recall']:.6f} precision={row['precision']:.6f} avg_hits={row['avg_hits']:.3f} waste={row['wasted_prefetch_ratio']:.6f}",flush=True)
    print(f'json_saved={a.output.with_suffix(".json")}',flush=True); print(f'csv_saved={a.output.with_suffix(".csv")}',flush=True)
    print('======================================================',flush=True)
if __name__=='__main__': main()
