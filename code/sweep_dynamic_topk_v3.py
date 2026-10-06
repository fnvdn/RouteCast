"""Calibrated mass -> dynamic Top-K Pareto sweep for RouteCast V3."""
from __future__ import annotations
import argparse,csv,json,time
from pathlib import Path
import torch
from routecast_rc.universal_model_v3 import UniversalRouteCastV3
from calibrate_routecast_v3 import batches

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--cache',type=Path,required=True); ap.add_argument('--model',type=Path,required=True); ap.add_argument('--calibration',type=Path,required=True); ap.add_argument('--save',type=Path,required=True)
    ap.add_argument('--masses',type=float,nargs='+',default=[i/100 for i in range(1,101)]); ap.add_argument('--max-k',type=int,default=32); ap.add_argument('--batch-size',type=int,default=1024); a=ap.parse_args()
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); man=json.loads((a.cache/'manifest.json').read_text(encoding='utf-8')); items=man['items']; cal=json.loads(a.calibration.read_text(encoding='utf-8'))
    ck=torch.load(a.model,map_location='cpu'); net=UniversalRouteCastV3(ck['history'],disabled_branches=ck.get('disabled_branches',[])).to(device); net.load_state_dict(ck['model']); net.eval()
    report={'method':'calibrated cumulative-mass dynamic Top-K','masses':a.masses,'max_k':a.max_k,'models':{}}
    print(f'device={device} predictor_frozen=True max_k={a.max_k}',flush=True)
    with torch.inference_mode():
        for model,spec in man['models'].items():
            temp=float(cal['temperatures'][model]); subset=[x for x in items if x['model']==model and x['split']=='test']; sums={m:[0.,0.,0.,0.] for m in a.masses}; start=time.time(); examples=0
            fixed_hit=fixed_actual=fixed_pred=0.
            for fi,it in enumerate(subset,1):
                d=torch.load(a.cache/it['file'],map_location='cpu')
                for static,hist,ctx,y in batches(d,a.batch_size,device):
                    with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=='cuda'): logits,_=net(static,hist,ctx)
                    # Cumulative score mass is defined on unit-sum ranking
                    # weights. For multi-label routing these weights are not
                    # interpreted as categorical probabilities. Temperature
                    # does not change Top-K.
                    p=torch.softmax(logits.float()/temp,dim=1); v,idx=p.sort(1,descending=True); cap=min(a.max_k,p.shape[1]); chosen=idx[:,:cap]; truth=y.gather(1,chosen); cdf=v[:,:cap].cumsum(1); actual=y.sum().item(); examples+=len(y)
                    native=min(spec['native_topk'],p.shape[1]); fixed_hit+=truth[:,:native].sum().item(); fixed_actual+=actual; fixed_pred+=len(y)*native
                    for mass in a.masses:
                        k=(cdf<mass).sum(1)+1; k=k.clamp(max=cap); mask=torch.arange(cap,device=device)[None,:]<k[:,None]; hit=(truth*mask).sum().item(); pred=k.sum().item(); x=sums[mass]; x[0]+=hit; x[1]+=actual; x[2]+=pred; x[3]+=len(y)
                if fi==1 or fi%10==0 or fi==len(subset): print(f'{model} test_file={fi}/{len(subset)} examples={examples} elapsed={time.time()-start:.1f}s',flush=True)
            policies={}
            for mass,(hit,actual,pred,n) in sums.items(): policies[f'{mass:.2f}']={'avg_k':pred/n,'recall':hit/max(actual,1),'precision':hit/max(pred,1),'waste_rate':1-hit/max(pred,1)}
            report['models'][model]={'temperature':temp,'native_topk':spec['native_topk'],'fixed_native':{'avg_k':spec['native_topk'],'recall':fixed_hit/max(fixed_actual,1),'precision':fixed_hit/max(fixed_pred,1)},'policies':policies}
            print(f'\n{model} T={temp:.4f} fixed_K={spec["native_topk"]} fixed_recall={fixed_hit/max(fixed_actual,1):.6f}',flush=True)
            for mass,x in policies.items(): print(f'  mass={mass} avg_k={x["avg_k"]:.4f} recall={x["recall"]:.6f} precision={x["precision"]:.6f} waste={x["waste_rate"]:.6f}',flush=True)
    a.save.parent.mkdir(parents=True,exist_ok=True); a.save.write_text(json.dumps(report,indent=2),encoding='utf-8'); csv_path=a.save.with_suffix('.csv')
    with csv_path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f); w.writerow(['model','temperature','policy','mass','avg_k','recall','precision','waste_rate'])
        for model,x in report['models'].items():
            n=x['fixed_native']; w.writerow([model,x['temperature'],'fixed_native','',n['avg_k'],n['recall'],n['precision'],1-n['precision']])
            for mass,m in x['policies'].items(): w.writerow([model,x['temperature'],'dynamic_mass',mass,m['avg_k'],m['recall'],m['precision'],m['waste_rate']])
    print(f'\nreport_saved={a.save}\nsummary_saved={csv_path}',flush=True)
if __name__=='__main__': main()
