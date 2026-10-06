"""Validation-tuned request-aware heatmap baseline on RouteCast caches."""
from __future__ import annotations
import argparse,json,math,time
from collections import defaultdict
from pathlib import Path
import torch

BUDGETS={'qwen3':[8,12,16],'deepseek_r1':[8,12,16],'llama4_maverick':[1,2,4,8]}

def unpack_binary(x,width):
    shifts=torch.arange(8,dtype=torch.uint8)
    return (((x.unsqueeze(-1)>>shifts)&1).reshape(*x.shape[:-1],-1))[...,:width]

def batches(data,batch,device):
    n=len(data['labels']); width=int(data.get('_num_experts',data['static'].shape[1]))
    for start in range(0,n,batch):
        static=data['static'][start:start+batch].float().to(device,non_blocking=True)
        labels=data['labels'][start:start+batch]
        if data.get('_binary_packed',False): labels=unpack_binary(labels,width)
        yield static,labels.float().to(device,non_blocking=True)

def validation_search(items,root,model,native_k,alphas,batch,device):
    hits={a:0.0 for a in alphas}; actual=0.0; start=time.time()
    selected=[x for x in items if x['model']==model and x['split']=='val']
    with torch.inference_mode():
        for fi,item in enumerate(selected,1):
            data=torch.load(root/item['file'],map_location='cpu')
            for static,y in batches(data,batch,device):
                one=static[:,:,1]; request=static[:,:,3]; actual+=y.sum().item()
                for alpha in alphas:
                    score=alpha*one+(1-alpha)*request
                    index=score.topk(min(native_k,score.shape[1]),1).indices
                    hits[alpha]+=y.gather(1,index).sum().item()
            if fi==1 or fi%10==0 or fi==len(selected): print(f'{model} val_progress={fi}/{len(selected)} elapsed={time.time()-start:.1f}s',flush=True)
    recalls={a:h/max(actual,1) for a,h in hits.items()}; best=max(alphas,key=lambda a:(recalls[a],-abs(a-.5)))
    return best,recalls

def test(items,root,model,alpha,budgets,batch,device):
    hit=defaultdict(float); pred=defaultdict(float); rr=defaultdict(float); ndcg=defaultdict(float); actual=examples=0; start=time.time()
    selected=[x for x in items if x['model']==model and x['split']=='test']
    with torch.inference_mode():
        for fi,item in enumerate(selected,1):
            data=torch.load(root/item['file'],map_location='cpu')
            for static,y in batches(data,batch,device):
                score=alpha*static[:,:,1]+(1-alpha)*static[:,:,3]
                actual+=y.sum().item(); examples+=len(y)
                truth=y.bool(); best_true=score.masked_fill(~truth,float('-inf')).amax(1,keepdim=True)
                first_rank=1+(score>best_true).sum(1); batch_rr=torch.where(truth.sum(1)>0,first_rank.float().reciprocal(),torch.zeros_like(first_rank,dtype=torch.float))
                for k in budgets:
                    kk=min(k,score.shape[1]); index=score.topk(kk,1).indices; rel=y.gather(1,index).float()
                    hit[k]+=rel.sum().item(); pred[k]+=len(y)*kk; rr[k]+=batch_rr.sum().item()
                    discounts=1/torch.log2(torch.arange(2,kk+2,device=device,dtype=torch.float32)); dcg=(rel*discounts).sum(1)
                    ideal_n=torch.minimum(truth.sum(1),torch.tensor(kk,device=device)); prefix=torch.cat([torch.zeros(1,device=device),discounts.cumsum(0)])
                    idcg=prefix[ideal_n]; ndcg[k]+=torch.where(idcg>0,dcg/idcg,torch.zeros_like(dcg)).sum().item()
            if fi==1 or fi%10==0 or fi==len(selected): print(f'{model} test_progress={fi}/{len(selected)} elapsed={time.time()-start:.1f}s',flush=True)
    return {str(k):{'recall':hit[k]/max(actual,1),'precision':hit[k]/max(pred[k],1),'mrr':rr[k]/max(examples,1),'ndcg':ndcg[k]/max(examples,1)} for k in budgets},examples

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--cache',type=Path,required=True); ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--alpha-step',type=float,default=.05); ap.add_argument('--batch-size',type=int,default=2048); a=ap.parse_args()
    manifest=json.loads((a.cache/'manifest.json').read_text(encoding='utf-8')); items=manifest['items']; device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    count=round(1/a.alpha_step); alphas=[round(i*a.alpha_step,10) for i in range(count+1)]
    report={'method':'request_aware_heatmap','formula':'alpha * one_step_heatmap + (1-alpha) * prefill_distribution','alpha_step':a.alpha_step,'models':{}}
    print(f'device={device} alpha_candidates={len(alphas)}',flush=True)
    for model,spec in manifest['models'].items():
        native=spec['native_topk']; best,val=validation_search(items,a.cache,model,native,alphas,a.batch_size,device)
        metrics,examples=test(items,a.cache,model,best,BUDGETS[model],a.batch_size,device)
        report['models'][model]={'native_topk':native,'best_alpha':best,'val_native_recall':val[best],'test_examples':examples,'test':metrics}
        print(f'{model} best_alpha={best:.2f} val_recall={val[best]:.6f}',flush=True)
        for k,x in metrics.items(): print(f'{model:18s} K={int(k):2d} recall={x["recall"]:.6f} precision={x["precision"]:.6f}',flush=True)
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(f'report_saved={a.output}',flush=True)
if __name__=='__main__': main()
