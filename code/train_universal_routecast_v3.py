"""Multi-budget RouteCast V3 training with cross-model ablation modes."""
from __future__ import annotations
import argparse,copy,json,math,random,time
from collections import defaultdict
from pathlib import Path
import torch
from torch import nn
from routecast_rc.universal_model_v3 import UniversalRouteCastV3

BUDGETS={'qwen3':[8,12,16],'deepseek_r1':[8,12,16],'llama4_maverick':[1,2,4,8]}

def rank_loss(scores,y,nneg=24):
    scores=scores.float(); y=y.float()
    pos=y.bool(); n=min(nneg,scores.shape[1]-1); neg=scores.masked_fill(pos,-1e9).topk(n,1).values
    np=max(1,int(y.sum(1).max())); ps=scores.masked_fill(~pos,-1e9).topk(np,1).values; valid=(ps>-1e8).float()
    return (nn.functional.softplus(-(ps.unsqueeze(2)-neg.unsqueeze(1)))*valid.unsqueeze(2)).sum()/(valid.sum()*n+1e-8)

def budget_loss(scores,y,ks):
    scores=scores.float(); y=y.float()
    pos=y.bool(); ps=scores.masked_fill(~pos,-1e9); valid=(ps>-1e8).float(); total=0
    for k in ks:
        threshold=scores.topk(min(k,scores.shape[1]),1).values[:,-1:]
        total+=(nn.functional.softplus(threshold-ps)*valid).sum()/(valid.sum()+1e-8)
    return total/len(ks)

def objective(scores,y,ks):
    # All reductions are deliberately evaluated in FP32.  With millions of
    # samples, FP16 reductions occasionally overflowed and contaminated the
    # Group-DRO weights even though the selected checkpoint remained usable.
    scores=scores.float(); y=y.float(); k=y.sum(1).mean().clamp_min(1); pw=((scores.shape[1]-k)/k).detach()
    bce=nn.functional.binary_cross_entropy_with_logits(scores,y,pos_weight=pw)
    target=y/y.sum(1,keepdim=True).clamp_min(1)
    listwise=(-(target*scores.log_softmax(1)).sum(1).mean())/math.log(scores.shape[1])
    return .20*bce+.25*listwise+.35*rank_loss(scores,y)+.20*budget_loss(scores,y,ks)

def unpack_binary(tensor,width):
    shifts=torch.arange(8,dtype=torch.uint8)
    bits=((tensor.unsqueeze(-1)>>shifts)&1).reshape(*tensor.shape[:-1],-1)
    return bits[...,:width]

def iter_batches(d,batch,dev,shuffle=True):
    n=len(d['labels']); order=torch.randperm(n) if shuffle else torch.arange(n)
    for st in range(0,n,batch):
        ix=order[st:st+batch]
        static=d['static'][ix].float(); history=d['history'][ix]; context=d['context'][ix].float(); labels=d['labels'][ix]
        if d.get('_binary_packed',False):
            width=int(d.get('_num_experts',static.shape[1])); history=unpack_binary(history,width); labels=unpack_binary(labels,width)
        yield tuple(x.float().to(dev,non_blocking=True) for x in (static,history,context,labels))

def amp_dtype(dev):
    # BF16 has FP32-like exponent range and is substantially safer than FP16
    # for the log-sum-exp/ranking objective. RTX 30/40-series GPUs support it.
    return torch.bfloat16 if dev.type=='cuda' and torch.cuda.is_bf16_supported() else torch.float16

def assert_finite(x,label,**where):
    if not torch.isfinite(x).all():
        bad=(~torch.isfinite(x)).sum().item()
        finite=x[torch.isfinite(x)]
        lo=float(finite.min()) if finite.numel() else float('nan')
        hi=float(finite.max()) if finite.numel() else float('nan')
        location=' '.join(f'{k}={v}' for k,v in where.items())
        raise FloatingPointError(f'nonfinite {label}: count={bad} finite_range=[{lo},{hi}] {location}')

def evaluate(net,items,manifest,root,split,batch,dev,model_names=None,progress=True):
    out={}; net.eval(); start=time.time()
    selected=list(manifest['models']) if model_names is None else list(model_names)
    with torch.inference_mode():
        for name in selected:
            spec=manifest['models'][name]
            subset=[x for x in items if x['split']==split and x['model']==name]; ks=BUDGETS[name]
            hit=defaultdict(float); pred=defaultdict(float); actual=examples=0
            for fi,it in enumerate(subset,1):
                d=torch.load(root/it['file'],map_location='cpu')
                for static,hist,ctx,y in iter_batches(d,batch,dev,False):
                    with torch.autocast(device_type=dev.type,dtype=amp_dtype(dev),enabled=dev.type=='cuda'): scores,_=net(static,hist,ctx)
                    assert_finite(scores,'evaluation_scores',split=split,model=name,file=fi)
                    actual+=y.sum().item(); examples+=len(y)
                    for k in ks:
                        kk=min(k,scores.shape[1]); idx=scores.topk(kk,1).indices; hit[k]+=y.gather(1,idx).sum().item(); pred[k]+=len(y)*kk
                if progress and (fi==1 or fi%10==0 or fi==len(subset)): print(f'{split}_progress model={name} file={fi}/{len(subset)} elapsed={time.time()-start:.1f}s',flush=True)
            out[name]={'examples':examples,'native_topk':spec['native_topk'],'metrics':{str(k):{'recall':hit[k]/max(actual,1),'precision':hit[k]/max(pred[k],1)} for k in ks}}
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--cache',type=Path,required=True); ap.add_argument('--epochs',type=int,default=6)
    ap.add_argument('--batch-size',type=int,default=256); ap.add_argument('--lr',type=float,default=6e-4); ap.add_argument('--dro-eta',type=float,default=.8)
    ap.add_argument('--save',type=Path,required=True); ap.add_argument('--seed',type=int,default=2026)
    ap.add_argument('--max-train-files',type=int,default=0,help='Debug/regression only: cap train files after shuffling; 0 uses all')
    ap.add_argument('--max-batches-per-file',type=int,default=0,help='Debug/regression only: cap batches per file; 0 uses all')
    ap.add_argument('--baseline-report',type=Path,default=None,help='Optional paper-heatmap JSON for direct comparison')
    ap.add_argument('--disable-branches',nargs='*',default=[],choices=UniversalRouteCastV3.BRANCH_NAMES)
    ap.add_argument('--models',nargs='+',default=None,
                    help='Model families to train/evaluate. One name gives a per-model model.')
    ap.add_argument('--disable-group-dro',action='store_true',
                    help='Keep static sample balancing but hold group weights uniform.')
    a=ap.parse_args()
    random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed); dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    m=json.loads((a.cache/'manifest.json').read_text(encoding='utf-8')); items=m['items']; available=list(m['models'])
    names=available if a.models is None else list(dict.fromkeys(a.models))
    unknown=[n for n in names if n not in m['models']]
    if unknown: raise ValueError(f'unknown model names: {unknown}; available={available}')
    if not names: raise ValueError('at least one model is required')
    train=[x for x in items if x['split']=='train' and x['model'] in names]
    totals=defaultdict(int)
    for x in train: totals[x['model']]+=x['samples']
    train_names=[n for n in names if totals[n]>0]; alln=sum(totals.values())
    balance={n:alln/(len(train_names)*totals[n]) for n in train_names}; q={n:1/len(train_names) for n in train_names}
    net=UniversalRouteCastV3(m['history'],disabled_branches=a.disable_branches).to(dev); opt=torch.optim.AdamW(net.parameters(),lr=a.lr,weight_decay=1e-4)
    scaler=torch.amp.GradScaler('cuda',enabled=dev.type=='cuda' and amp_dtype(dev)==torch.float16)
    training_mode='per_model' if len(names)==1 else ('shared_no_dro' if a.disable_group_dro else 'shared_dro')
    print(f'device={dev} amp_dtype={amp_dtype(dev)} grad_scaler={scaler.is_enabled()} training_mode={training_mode} models={names} train_files={len(train)} samples={dict(totals)} disabled_branches={a.disable_branches}',flush=True)
    best=-1; best_state=None; best_epoch=0
    for ep in range(a.epochs):
        net.train(); random.shuffle(train); epoch_train=train[:a.max_train_files] if a.max_train_files>0 else train; sums=defaultdict(float); counts=defaultdict(int); seen=0; start=time.time()
        for fi,it in enumerate(epoch_train,1):
            d=torch.load(a.cache/it['file'],map_location='cpu'); name=it['model']
            for bi,(static,hist,ctx,y) in enumerate(iter_batches(d,a.batch_size,dev,True),1):
                if a.max_batches_per_file>0 and bi>a.max_batches_per_file: break
                assert_finite(static,'static',epoch=ep+1,file=fi,batch=bi,model=name)
                assert_finite(hist,'history',epoch=ep+1,file=fi,batch=bi,model=name)
                assert_finite(ctx,'context',epoch=ep+1,file=fi,batch=bi,model=name)
                assert_finite(y,'labels',epoch=ep+1,file=fi,batch=bi,model=name)
                with torch.autocast(device_type=dev.type,dtype=amp_dtype(dev),enabled=dev.type=='cuda'):
                    scores,_=net(static,hist,ctx); raw=objective(scores,y,BUDGETS[name]); loss=raw*balance[name]*q[name]*len(train_names)
                assert_finite(scores,'scores',epoch=ep+1,file=fi,batch=bi,model=name)
                assert_finite(raw,'raw_loss',epoch=ep+1,file=fi,batch=bi,model=name)
                assert_finite(loss,'weighted_loss',epoch=ep+1,file=fi,batch=bi,model=name)
                opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
                grad_norm=nn.utils.clip_grad_norm_(net.parameters(),1.0,error_if_nonfinite=True)
                assert_finite(grad_norm,'gradient_norm',epoch=ep+1,file=fi,batch=bi,model=name)
                scaler.step(opt); scaler.update()
                sums[name]+=raw.item()*len(y); counts[name]+=len(y); seen+=len(y)
            if fi==1 or fi%5==0 or fi==len(epoch_train):
                mem=torch.cuda.max_memory_allocated()/1e6 if dev.type=='cuda' else 0
                print(f'epoch={ep+1}/{a.epochs} file={fi}/{len(epoch_train)} model={name} samples={seen} loss={loss.item():.6f} grad_norm={grad_norm.item():.4f} gpu_mb={mem:.1f} elapsed={time.time()-start:.1f}s',flush=True)
        means={n:sums[n]/max(counts[n],1) for n in train_names}
        if not all(math.isfinite(v) for v in means.values()): raise FloatingPointError(f'nonfinite group means epoch={ep+1}: {means}')
        # Stable float64 softmax avoids overflow and makes the update auditable.
        # The no-DRO control keeps q uniform while retaining identical static
        # sample-count balancing and all other optimization settings.
        if not a.disable_group_dro and len(train_names)>1:
            dro_logits=[math.log(max(q[n],1e-12))+a.dro_eta*means[n] for n in train_names]; mx=max(dro_logits)
            exps=[math.exp(v-mx) for v in dro_logits]; den=sum(exps); q={n:exps[i]/den for i,n in enumerate(train_names)}
        else:
            q={n:1/len(train_names) for n in train_names}
        if not all(math.isfinite(v) and v>0 for v in q.values()): raise FloatingPointError(f'nonfinite group weights epoch={ep+1}: {q}')
        print(f'group_weights epoch={ep+1} dro_enabled={not a.disable_group_dro and len(train_names)>1} losses={means} weights={q} weight_sum={sum(q.values()):.12f}',flush=True)
        val=evaluate(net,items,m,a.cache,'val',a.batch_size,dev,names,False)
        macro=sum(val[n]['metrics'][str(m['models'][n]['native_topk'])]['recall'] for n in names)/len(names)
        print(f'epoch={ep+1} val_macro_native_recall={macro:.6f}',flush=True)
        if macro>best: best=macro; best_epoch=ep+1; best_state=copy.deepcopy(net.state_dict()); print(f'best_checkpoint=epoch{best_epoch}',flush=True)
        # Recoverable epoch checkpoint for long full-scale runs.
        a.save.parent.mkdir(parents=True,exist_ok=True)
        torch.save({'model':net.state_dict(),'optimizer':opt.state_dict(),'epoch':ep+1,'history':m['history'],'group_weights':q,'best_epoch':best_epoch,'best_score':best,'best_model':best_state},a.save.with_suffix('.last.pt'))
        print(f'epoch_checkpoint_saved={a.save.with_suffix(".last.pt")}',flush=True)
    net.load_state_dict(best_state); val=evaluate(net,items,m,a.cache,'val',a.batch_size,dev,names); test=evaluate(net,items,m,a.cache,'test',a.batch_size,dev,names)
    report={'version':3,'task':m.get('task','cross_layer'),'training_mode':training_mode,'selected_models':names,'group_dro_enabled':not a.disable_group_dro and len(names)>1,'static_sample_balancing':True,'seed':a.seed,'history':m['history'],'disabled_branches':a.disable_branches,'best_epoch':best_epoch,'group_weights':q,'val':val,'test':test}
    a.save.parent.mkdir(parents=True,exist_ok=True); torch.save({'model':net.state_dict(),'history':m['history'],'disabled_branches':a.disable_branches,'version':3,'report':report},a.save); a.save.with_suffix('.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('\n================ ROUTECAST V3 SUMMARY ================',flush=True)
    for name in names:
        for k in BUDGETS[name]:
            x=test[name]['metrics'][str(k)]; print(f'{name:18s} K={k:2d} recall={x["recall"]:.6f} precision={x["precision"]:.6f}',flush=True)
    print(f'best_epoch={best_epoch} val_macro_native_recall={best:.6f}',flush=True); print(f'model_saved={a.save}',flush=True); print(f'report_saved={a.save.with_suffix(".json")}',flush=True)
    if a.baseline_report is not None:
        baseline=json.loads(a.baseline_report.read_text(encoding='utf-8'))
        key='cross_token' if m.get('task')=='cross_token' else 'cross_layer'
        print('\n============== PAPER HEATMAP COMPARISON ==============',flush=True)
        for name in names:
            k=str(m['models'][name]['native_topk']); ours=test[name]['metrics'][k]['recall']
            paper=baseline['models'][name][key][k]['recall']; delta=ours-paper
            print(f'{name:18s} K={int(k):2d} paper={paper:.6f} routecast={ours:.6f} delta={delta:+.6f} exceeded={delta>0}',flush=True)
    print('======================================================',flush=True)
if __name__=='__main__': main()
