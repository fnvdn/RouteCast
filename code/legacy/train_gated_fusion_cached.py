"""Train and evaluate the gating MLP from reusable cached features."""
import argparse,json,time,random
from pathlib import Path
import torch
from torch import nn

def augment_features(base, row_offset=0):
    """Expand [B,128,3] scores into per-expert + token-layer context features."""
    b,n,_=base.shape; pop,trans,gru=base[:,:,0],base[:,:,1],base[:,:,2]
    ranks=[]
    for score in (pop,trans,gru):
        order=score.argsort(1,descending=True); rank=torch.empty_like(order); rank.scatter_(1,order,torch.arange(n,device=base.device).expand(b,n)); ranks.append(rank.float()/(n-1))
    eps=1e-8
    gp=gru/(gru.sum(1,keepdim=True)+eps); tp=trans/(trans.sum(1,keepdim=True)+eps)
    gru_entropy=-(gp*(gp+eps).log()).sum(1)/torch.log(torch.tensor(float(n),device=base.device)); trans_entropy=-(tp*(tp+eps).log()).sum(1)/torch.log(torch.tensor(float(n),device=base.device))
    gv=gru.topk(2,1).values; gru_max=gv[:,0]; gru_margin=gv[:,0]-gv[:,1]
    masks=[]
    for score in (pop,trans,gru):
        m=torch.zeros_like(score,dtype=torch.bool); m.scatter_(1,score.topk(8,1).indices,True); masks.append(m)
    overlaps=[(masks[0]&masks[1]).sum(1).float()/8,(masks[0]&masks[2]).sum(1).float()/8,(masks[1]&masks[2]).sum(1).float()/8]
    layer=((torch.arange(b,device=base.device)+row_offset)%93).float()/92
    context=torch.stack([gru_max,gru_margin,gru_entropy,trans_entropy,*overlaps,layer],1)
    return torch.cat([base,*[r.unsqueeze(2) for r in ranks],context.unsqueeze(1).expand(-1,n,-1)],2)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--cache',type=Path,required=True); ap.add_argument('--epochs',type=int,default=10); ap.add_argument('--batch-size',type=int,default=8192); ap.add_argument('--save',type=Path,required=True); ap.add_argument('--ablation',choices=['full','no_gru','no_transition','no_popularity'],default='full'); ap.add_argument('--seed',type=int,default=2026); a=ap.parse_args(); random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed); dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); m=json.loads((a.cache/'manifest.json').read_text(encoding='utf-8')); items=m['items']; print(f'device={dev} cached_files={len(items)} ablation={a.ablation} seed={a.seed}',flush=True)
    input_dim=14; net=nn.Sequential(nn.Linear(input_dim,64),nn.ReLU(),nn.Linear(64,32),nn.ReLU(),nn.Linear(32,1)).to(dev); opt=torch.optim.AdamW(net.parameters(),lr=1e-3); lossfn=nn.BCEWithLogitsLoss(); print(f'gate_input_dim={input_dim}',flush=True)
    for ep in range(a.epochs):
        net.train(); total=n=0; start=time.time()
        train_items=[x for x in items if x['split']=='train']
        for i,it in enumerate(train_items,1):
            d=torch.load(a.cache/it['file'],map_location='cpu'); f=d['features'].float(); y=d['labels'].float()
            for st in range(0,len(y),a.batch_size):
                fb=f[st:st+a.batch_size].clone();
                if a.ablation=='no_gru': fb[:,:,2]=0
                elif a.ablation=='no_transition': fb[:,:,1]=0
                elif a.ablation=='no_popularity': fb[:,:,0]=0
                xb=augment_features(fb.to(dev),st); yb=y[st:st+a.batch_size].to(dev); loss=lossfn(net(xb).squeeze(2),yb); opt.zero_grad(); loss.backward(); opt.step(); total+=loss.item()*len(yb); n+=len(yb)
            print(f'epoch={ep+1}/{a.epochs} file={i}/{len(train_items)} loss={total/n:.6f} elapsed={time.time()-start:.1f}s',flush=True)
    def evaluate(kind):
        hit=actual=pred=0; subset=[x for x in items if x['split']==kind]; net.eval(); start=time.time()
        with torch.no_grad():
            for i,it in enumerate(subset,1):
                d=torch.load(a.cache/it['file'],map_location='cpu'); f=d['features'].float(); y=d['labels'];
                for st in range(0,len(y),a.batch_size):
                    yb=y[st:st+a.batch_size]; fb=f[st:st+a.batch_size].clone();
                    if a.ablation=='no_gru': fb[:,:,2]=0
                    elif a.ablation=='no_transition': fb[:,:,1]=0
                    elif a.ablation=='no_popularity': fb[:,:,0]=0
                    xb=augment_features(fb.to(dev),st); top=net(xb).squeeze(2).topk(8,1).indices.cpu(); hit+=yb.gather(1,top).sum().item(); actual+=yb.sum().item(); pred+=len(yb)*8
                print(f'{kind}_progress={i}/{len(subset)} elapsed={time.time()-start:.1f}s',flush=True)
        return hit/actual,hit/pred
    vr,vp=evaluate('val'); tr,tp=evaluate('test'); print(f'val_recall@8={vr:.6f} val_precision@8={vp:.6f}'); print(f'test_recall@8={tr:.6f} test_precision@8={tp:.6f}'); a.save.parent.mkdir(parents=True,exist_ok=True); torch.save({'gate':net.state_dict(),'history':m['history'],'input_dim':input_dim,'ablation':a.ablation,'seed':a.seed,'metrics':{'val_recall@8':vr,'val_precision@8':vp,'test_recall@8':tr,'test_precision@8':tp}},a.save); print(f'model_saved={a.save}')
if __name__=='__main__': main()
