"""Train RouteCast with multi-K ranking and prefetch-cost regularization.

Input cache format is the existing fusion cache (manifest.json plus *.pt files
containing ``features`` [N,128,3] and ``labels`` [N,128]).
"""
import argparse, json, random, time
from pathlib import Path
import torch
from torch import nn

from legacy.train_gated_fusion_cached import augment_features


def pairwise_rank_loss(scores, labels, negatives=16):
    pos = labels.bool()
    # hardest negatives make the loss ranking-aware without an O(E^2) tensor.
    neg_scores = scores.masked_fill(pos, -1e9)
    neg = neg_scores.topk(min(negatives, scores.shape[1]), dim=1).values
    pos_scores = scores.masked_fill(~pos, -1e9).topk(min(8, scores.shape[1]), dim=1).values
    valid = (pos_scores > -1e8).float()
    return (torch.nn.functional.softplus(-(pos_scores.unsqueeze(-1)-neg.unsqueeze(1))) * valid.unsqueeze(-1)).sum() / (valid.sum()*neg.shape[-1]+1e-6)


def topk_cost_loss(scores, labels, ks):
    # Differentiable approximation: penalize probability mass outside true Top-8
    p = scores.sigmoid()
    loss = 0.0
    for k in ks:
        idx = scores.topk(min(k, scores.shape[1]), 1).indices
        picked = labels.gather(1, idx).float()
        loss = loss + (1.0 - picked.mean(1)).mean() / float(k)
    return loss / len(ks)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--cache',type=Path,required=True); ap.add_argument('--epochs',type=int,default=3)
    ap.add_argument('--batch-size',type=int,default=8192); ap.add_argument('--save',type=Path,required=True)
    ap.add_argument('--lr',type=float,default=1e-3); ap.add_argument('--rank-weight',type=float,default=.35)
    ap.add_argument('--cost-weight',type=float,default=.05); ap.add_argument('--ks',default='1,2,4,8,16')
    ap.add_argument('--seed',type=int,default=2026); a=ap.parse_args()
    random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    manifest=json.loads((a.cache/'manifest.json').read_text(encoding='utf-8')); items=manifest['items']
    ks=tuple(int(x) for x in a.ks.split(',')); train=[x for x in items if x['split']=='train']
    print(f'device={dev} train_files={len(train)} epochs={a.epochs} ks={ks}',flush=True)
    net=nn.Sequential(nn.Linear(14,64),nn.ReLU(),nn.Linear(64,32),nn.ReLU(),nn.Linear(32,1)).to(dev)
    opt=torch.optim.AdamW(net.parameters(),lr=a.lr); bce=nn.BCEWithLogitsLoss()
    for ep in range(a.epochs):
        net.train(); total=n=0; start=time.time()
        for fi,it in enumerate(train,1):
            d=torch.load(a.cache/it['file'],map_location='cpu'); f=d['features'].float(); y=d['labels'].float()
            for st in range(0,len(y),a.batch_size):
                fb=f[st:st+a.batch_size].to(dev); yb=y[st:st+a.batch_size].to(dev)
                z=net(augment_features(fb,st)).squeeze(-1)
                loss=bce(z,yb)+a.rank_weight*pairwise_rank_loss(z,yb)+a.cost_weight*topk_cost_loss(z,yb,ks)
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); total+=loss.item()*len(yb); n+=len(yb)
            print(f'epoch={ep+1}/{a.epochs} file={fi}/{len(train)} samples={n} loss={total/n:.6f} elapsed={time.time()-start:.1f}s',flush=True)
    def evaluate(split):
        hit={k:0.0 for k in ks}; denom=0.0; subset=[x for x in items if x['split']==split]; net.eval(); start=time.time()
        with torch.no_grad():
            for fi,it in enumerate(subset,1):
                d=torch.load(a.cache/it['file'],map_location='cpu'); f=d['features'].float(); y=d['labels'].float()
                for st in range(0,len(y),a.batch_size):
                    z=net(augment_features(f[st:st+a.batch_size].to(dev),st)).squeeze(-1); yy=y[st:st+a.batch_size].to(dev)
                    for k in ks: hit[k]+=yy.gather(1,z.topk(min(k,z.shape[1]),1).indices).sum().item()
                    denom+=yy.sum().item()
                print(f'{split}_progress={fi}/{len(subset)} elapsed={time.time()-start:.1f}s',flush=True)
        return {f'recall@{k}':hit[k]/max(denom,1.0) for k in ks}
    val=evaluate('val'); test=evaluate('test')
    a.save.parent.mkdir(parents=True,exist_ok=True)
    report={'cache':str(a.cache),'device':str(dev),'epochs':a.epochs,'batch_size':a.batch_size,'ks':ks,'rank_weight':a.rank_weight,'cost_weight':a.cost_weight,'val':val,'test':test}
    torch.save({'gate':net.state_dict(),'input_dim':14,'ks':ks,'seed':a.seed,'metrics':report},a.save)
    print('\n========== FINAL SUMMARY ==========',flush=True)
    print(f"device={dev} epochs={a.epochs} batch_size={a.batch_size} ks={ks}",flush=True)
    for k in ks: print(f"K={k} val_recall={val[f'recall@{k}']:.6f} test_recall={test[f'recall@{k}']:.6f}",flush=True)
    print(f'model_saved={a.save}',flush=True)
    print('===================================',flush=True)

if __name__=='__main__': main()
