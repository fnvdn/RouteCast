"""Per-model temperature calibration for a frozen RouteCast V3 predictor."""
from __future__ import annotations
import argparse,csv,json,time
from pathlib import Path
import torch
from torch import nn
from routecast_rc.universal_model_v3 import UniversalRouteCastV3

def unpack_binary(x,width):
    shifts=torch.arange(8,dtype=torch.uint8)
    return (((x.unsqueeze(-1)>>shifts)&1).reshape(*x.shape[:-1],-1))[...,:width]

def batches(d,batch,device):
    width=int(d.get('_num_experts',d['static'].shape[1])); n=len(d['labels'])
    for st in range(0,n,batch):
        static=d['static'][st:st+batch].float(); hist=d['history'][st:st+batch]; ctx=d['context'][st:st+batch].float(); y=d['labels'][st:st+batch]
        if d.get('_binary_packed',False): hist=unpack_binary(hist,width); y=unpack_binary(y,width)
        yield static.to(device),hist.float().to(device),ctx.to(device),y.float().to(device)

def collect(net,items,root,model,split,batch,device,limit=0):
    zs=[]; ys=[]; seen=0; start=time.time(); subset=[x for x in items if x['model']==model and x['split']==split]
    with torch.inference_mode():
        for fi,it in enumerate(subset,1):
            d=torch.load(root/it['file'],map_location='cpu')
            for static,hist,ctx,y in batches(d,batch,device):
                with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=='cuda'): z,_=net(static,hist,ctx)
                take=len(y) if not limit else min(len(y),max(0,limit-seen))
                if take: zs.append(z[:take].float().cpu()); ys.append(y[:take].to(torch.uint8).cpu()); seen+=take
                if limit and seen>=limit: break
            if fi==1 or fi%10==0 or fi==len(subset) or (limit and seen>=limit): print(f'{model} {split}_file={fi}/{len(subset)} examples={seen} elapsed={time.time()-start:.1f}s',flush=True)
            if limit and seen>=limit: break
    return torch.cat(zs),torch.cat(ys).float()

def fit_temperature(z,y,device,max_iter):
    z=z.to(device); y=y.to(device); log_t=nn.Parameter(torch.zeros((),device=device)); opt=torch.optim.LBFGS([log_t],lr=.25,max_iter=max_iter,line_search_fn='strong_wolfe')
    def closure():
        opt.zero_grad(); t=log_t.exp().clamp(.05,20); loss=nn.functional.binary_cross_entropy_with_logits(z/t,y); loss.backward(); return loss
    before=nn.functional.binary_cross_entropy_with_logits(z,y).item(); opt.step(closure); t=float(log_t.detach().exp().clamp(.05,20)); after=nn.functional.binary_cross_entropy_with_logits(z/t,y).item()
    return t,before,after

def metrics(z,y,t,native,bins=15):
    p=torch.sigmoid(z/t); bce=nn.functional.binary_cross_entropy_with_logits(z/t,y).item(); brier=((p-y)**2).mean().item()
    conf=p.flatten(); truth=y.flatten(); ece=0.; reliability=[]
    for i in range(bins):
        lo=i/bins; hi=(i+1)/bins; mask=(conf>=lo)&(conf<(hi if i<bins-1 else hi+1e-6)); count=int(mask.sum())
        if count: avg_conf=float(conf[mask].mean()); accuracy=float(truth[mask].mean()); ece+=count/len(conf)*abs(avg_conf-accuracy)
        else: avg_conf=accuracy=0.
        reliability.append({'lower':lo,'upper':hi,'count':count,'confidence':avg_conf,'accuracy':accuracy})
    idx=z.topk(min(native,z.shape[1]),1).indices; hits=y.gather(1,idx).sum().item(); recall=hits/max(y.sum().item(),1); precision=hits/(len(y)*native)
    return {'bce':bce,'brier':brier,'ece':ece,'recall_at_native_k':recall,'precision_at_native_k':precision,'reliability':reliability}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--cache',type=Path,required=True); ap.add_argument('--model',type=Path,required=True); ap.add_argument('--save',type=Path,required=True)
    ap.add_argument('--batch-size',type=int,default=1024); ap.add_argument('--max-calibration-examples',type=int,default=300000); ap.add_argument('--lbfgs-iterations',type=int,default=40); a=ap.parse_args()
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); manifest=json.loads((a.cache/'manifest.json').read_text(encoding='utf-8')); items=manifest['items']
    ckpt=torch.load(a.model,map_location='cpu'); net=UniversalRouteCastV3(ckpt['history'],disabled_branches=ckpt.get('disabled_branches',[])).to(device); net.load_state_dict(ckpt['model']); net.eval()
    report={'method':'per-model temperature scaling','checkpoint':str(a.model),'temperatures':{},'models':{}}
    print(f'device={device} predictor_frozen=True',flush=True)
    for model,spec in manifest['models'].items():
        zv,yv=collect(net,items,a.cache,model,'val',a.batch_size,device,a.max_calibration_examples); t,vb,va=fit_temperature(zv,yv,device,a.lbfgs_iterations); del zv,yv
        zt,yt=collect(net,items,a.cache,model,'test',a.batch_size,device,0); before=metrics(zt,yt,1.,spec['native_topk']); after=metrics(zt,yt,t,spec['native_topk']); del zt,yt
        report['temperatures'][model]=t; report['models'][model]={'native_topk':spec['native_topk'],'temperature':t,'val_bce_before':vb,'val_bce_after':va,'test_before':before,'test_after':after}
        print(f'{model:18s} T={t:.4f} val_bce={vb:.6f}->{va:.6f} test_bce={before["bce"]:.6f}->{after["bce"]:.6f} brier={before["brier"]:.6f}->{after["brier"]:.6f} ece={before["ece"]:.6f}->{after["ece"]:.6f} recall={before["recall_at_native_k"]:.6f}->{after["recall_at_native_k"]:.6f}',flush=True)
    a.save.parent.mkdir(parents=True,exist_ok=True); a.save.write_text(json.dumps(report,indent=2),encoding='utf-8')
    csv_path=a.save.with_suffix('.csv')
    with csv_path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f); w.writerow(['model','temperature','bce_before','bce_after','brier_before','brier_after','ece_before','ece_after','recall_before','recall_after'])
        for model,x in report['models'].items():
            b=x['test_before']; c=x['test_after']; w.writerow([model,x['temperature'],b['bce'],c['bce'],b['brier'],c['brier'],b['ece'],c['ece'],b['recall_at_native_k'],c['recall_at_native_k']])
    print(f'calibration_saved={a.save}\nsummary_saved={csv_path}',flush=True)
if __name__=='__main__': main()
