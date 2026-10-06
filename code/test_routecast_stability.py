"""Short, three-model forward/backward stability regression for RouteCast V3."""
from __future__ import annotations
import argparse,json,random,time
from pathlib import Path
import torch
from torch import nn
from routecast_rc.universal_model_v3 import UniversalRouteCastV3
from train_universal_routecast_v3 import BUDGETS,amp_dtype,assert_finite,iter_batches,objective

def main():
    p=argparse.ArgumentParser(); p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--batches-per-model',type=int,default=8); p.add_argument('--batch-size',type=int,default=256)
    p.add_argument('--seed',type=int,default=2026); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); m=json.loads((a.cache/'manifest.json').read_text(encoding='utf-8'))
    net=UniversalRouteCastV3(m['history']).to(dev); opt=torch.optim.AdamW(net.parameters(),lr=6e-4,weight_decay=1e-4)
    scaler=torch.amp.GradScaler('cuda',enabled=dev.type=='cuda' and amp_dtype(dev)==torch.float16)
    rows=[]; started=time.time()
    for name in m['models']:
        item=next(x for x in m['items'] if x['split']=='train' and x['model']==name); d=torch.load(a.cache/item['file'],map_location='cpu')
        losses=[]; gradients=[]
        for bi,(static,hist,ctx,y) in enumerate(iter_batches(d,a.batch_size,dev,True),1):
            if bi>a.batches_per_model: break
            with torch.autocast(device_type=dev.type,dtype=amp_dtype(dev),enabled=dev.type=='cuda'):
                scores,_=net(static,hist,ctx); loss=objective(scores,y,BUDGETS[name])
            assert_finite(scores,'scores',model=name,batch=bi); assert_finite(loss,'loss',model=name,batch=bi)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
            norm=nn.utils.clip_grad_norm_(net.parameters(),1.0,error_if_nonfinite=True); assert_finite(norm,'gradient_norm',model=name,batch=bi)
            scaler.step(opt); scaler.update(); losses.append(float(loss.detach())); gradients.append(float(norm.detach()))
            print(f'stability model={name} batch={bi}/{a.batches_per_model} loss={loss.item():.6f} grad_norm={norm.item():.6f}',flush=True)
        rows.append({'model':name,'batches':len(losses),'loss_min':min(losses),'loss_max':max(losses),'gradient_norm_min':min(gradients),'gradient_norm_max':max(gradients)})
    report={'status':'PASS','cache':str(a.cache),'seed':a.seed,'device':str(dev),'amp_dtype':str(amp_dtype(dev)),'rows':rows,'elapsed_seconds':time.time()-started}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(f'stability_status=PASS report_saved={a.output}',flush=True)

if __name__=='__main__': main()
