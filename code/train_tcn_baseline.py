"""Train the matched-protocol history-only TCN baseline."""
from __future__ import annotations
import argparse,copy,json,math,random,time
from collections import defaultdict
from pathlib import Path
import torch
from torch import nn
from routecast_rc.tcn_baseline import HistoryTCN
from train_universal_routecast_v3 import BUDGETS,amp_dtype,assert_finite,evaluate,iter_batches,objective

def main():
    p=argparse.ArgumentParser(); p.add_argument('--cache',type=Path,required=True); p.add_argument('--save',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=3); p.add_argument('--batch-size',type=int,default=256); p.add_argument('--channels',type=int,default=24,help='TCN hidden channels; 16 is safer on 8 GB GPUs.')
    p.add_argument('--lr',type=float,default=6e-4)
    p.add_argument('--dro-eta',type=float,default=.8); p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--max-train-files',type=int,default=0,help='Debug cap; 0 uses all training files.')
    p.add_argument('--max-batches-per-file',type=int,default=0,help='Debug cap; 0 uses every batch.')
    p.add_argument('--routecast-report',type=Path,default=None,help='Optional RouteCast JSON report for direct native-K comparison.')
    a=p.parse_args()
    random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed); dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    m=json.loads((a.cache/'manifest.json').read_text(encoding='utf-8')); items=m['items']; train=[x for x in items if x['split']=='train']; names=list(m['models'])
    totals=defaultdict(int)
    if a.max_train_files:
        selected=[]
        for name in names: selected.extend([x for x in train if x['model']==name][:a.max_train_files])
        train=selected
    for x in train: totals[x['model']]+=x['samples']
    alln=sum(totals.values()); balance={n:alln/(len(names)*totals[n]) for n in names}; q={n:1/len(names) for n in names}
    net=HistoryTCN(m['history'],channels=a.channels).to(dev); opt=torch.optim.AdamW(net.parameters(),lr=a.lr,weight_decay=1e-4)
    scaler=torch.amp.GradScaler('cuda',enabled=dev.type=='cuda' and amp_dtype(dev)==torch.float16); best=-1; state=None; best_epoch=0
    print(f'device={dev} amp_dtype={amp_dtype(dev)} parameters={sum(x.numel() for x in net.parameters())} train_files={len(train)} samples={dict(totals)}',flush=True)
    for ep in range(a.epochs):
        random.shuffle(train); net.train(); sums=defaultdict(float); counts=defaultdict(int); seen=0; started=time.time()
        for fi,item in enumerate(train,1):
            d=torch.load(a.cache/item['file'],map_location='cpu'); name=item['model']
            for bi,(static,hist,ctx,y) in enumerate(iter_batches(d,a.batch_size,dev,True),1):
                if a.max_batches_per_file and bi>a.max_batches_per_file: break
                with torch.autocast(device_type=dev.type,dtype=amp_dtype(dev),enabled=dev.type=='cuda'): scores,_=net(static,hist,ctx); raw=objective(scores,y,BUDGETS[name]); loss=raw*balance[name]*q[name]*len(names)
                assert_finite(scores,'scores',epoch=ep+1,file=fi,batch=bi,model=name); assert_finite(loss,'loss',epoch=ep+1,file=fi,batch=bi,model=name)
                opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt); norm=nn.utils.clip_grad_norm_(net.parameters(),1,error_if_nonfinite=True)
                scaler.step(opt); scaler.update(); sums[name]+=float(raw.detach())*len(y); counts[name]+=len(y); seen+=len(y)
            if fi==1 or fi%5==0 or fi==len(train): print(f'epoch={ep+1}/{a.epochs} file={fi}/{len(train)} model={name} samples={seen} loss={loss.item():.6f} grad_norm={norm.item():.4f} elapsed={time.time()-started:.1f}s',flush=True)
        means={n:sums[n]/counts[n] for n in names}; logits=[math.log(max(q[n],1e-12))+a.dro_eta*means[n] for n in names]; mx=max(logits); ex=[math.exp(x-mx) for x in logits]; q={n:ex[i]/sum(ex) for i,n in enumerate(names)}
        print(f'group_dro epoch={ep+1} losses={means} weights={q} weight_sum={sum(q.values()):.12f}',flush=True)
        val=evaluate(net,items,m,a.cache,'val',a.batch_size,dev,False); macro=sum(val[n]['metrics'][str(m['models'][n]['native_topk'])]['recall'] for n in names)/len(names)
        print(f'epoch={ep+1} val_macro_native_recall={macro:.6f}',flush=True)
        if macro>best: best=macro; best_epoch=ep+1; state=copy.deepcopy(net.state_dict()); print(f'best_checkpoint=epoch{best_epoch}',flush=True)
        a.save.parent.mkdir(parents=True,exist_ok=True)
        torch.save({'model':net.state_dict(),'model_type':'history_tcn','history':m['history'],'channels':a.channels,'epoch':ep+1,'best_epoch':best_epoch,'best_score':best,'best_model':state,'group_weights':q},a.save.with_suffix('.last.pt'))
        print(f'epoch_checkpoint_saved={a.save.with_suffix(".last.pt")}',flush=True)
    net.load_state_dict(state); val=evaluate(net,items,m,a.cache,'val',a.batch_size,dev); test=evaluate(net,items,m,a.cache,'test',a.batch_size,dev)
    report={'method':'history_only_tcn','seed':a.seed,'history':m['history'],'channels':a.channels,'parameters':sum(x.numel() for x in net.parameters()),'best_epoch':best_epoch,'group_weights':q,'val':val,'test':test}
    a.save.parent.mkdir(parents=True,exist_ok=True); torch.save({'model':net.state_dict(),'model_type':'history_tcn','history':m['history'],'channels':a.channels,'report':report},a.save); a.save.with_suffix('.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('\n================ TCN BASELINE SUMMARY ================',flush=True)
    for n in names:
        k=str(m['models'][n]['native_topk']); x=test[n]['metrics'][k]; print(f'{n:18s} K={int(k):2d} recall={x["recall"]:.6f} precision={x["precision"]:.6f}',flush=True)
    print(f'parameters={report["parameters"]}',flush=True)
    print(f'model_saved={a.save}\nreport_saved={a.save.with_suffix(".json")}',flush=True)
    if a.routecast_report is not None:
        routecast=json.loads(a.routecast_report.read_text(encoding='utf-8'))
        print('\n================ TCN VS ROUTECAST ================',flush=True)
        for n in names:
            k=str(m['models'][n]['native_topk'])
            tcn=test[n]['metrics'][k]['recall']
            rc=routecast['test'][n]['metrics'][k]['recall']
            print(f'{n:18s} K={int(k):2d} tcn={tcn:.6f} routecast={rc:.6f} delta={tcn-rc:+.6f}',flush=True)
        print('==================================================',flush=True)

if __name__=='__main__': main()
