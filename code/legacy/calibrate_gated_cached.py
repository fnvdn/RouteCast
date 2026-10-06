"""Streaming temperature scaling for cached gated fusion."""
import argparse,json,time
from pathlib import Path
import torch
from torch import nn

def augment(f,offset=0):
 b,n,_=f.shape; p,t,g=f[:,:,0],f[:,:,1],f[:,:,2]; rs=[]
 for s in (p,t,g):
  o=s.argsort(1,descending=True); r=torch.empty_like(o); r.scatter_(1,o,torch.arange(n).expand(b,n)); rs.append(r.float()/127)
 e=1e-8; gp=g/(g.sum(1,keepdim=True)+e); tp=t/(t.sum(1,keepdim=True)+e); ge=-(gp*(gp+e).log()).sum(1)/torch.log(torch.tensor(128.)); te=-(tp*(tp+e).log()).sum(1)/torch.log(torch.tensor(128.)); gv=g.topk(2,1).values; masks=[s.topk(8,1).indices for s in (p,t,g)]; ov=[(masks[0].unsqueeze(2)==masks[1].unsqueeze(1)).any(2).sum(1).float()/8,(masks[0].unsqueeze(2)==masks[2].unsqueeze(1)).any(2).sum(1).float()/8,(masks[1].unsqueeze(2)==masks[2].unsqueeze(1)).any(2).sum(1).float()/8]; ctx=torch.stack([gv[:,0],gv[:,0]-gv[:,1],ge,te,*ov,(torch.arange(b)%93).float()/92],1); return torch.cat([f,*[r.unsqueeze(2) for r in rs],ctx.unsqueeze(1).expand(-1,n,-1)],2)
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--cache',type=Path,required=True); ap.add_argument('--model',type=Path,required=True); ap.add_argument('--save',type=Path,required=True); ap.add_argument('--batch-size',type=int,default=8192); a=ap.parse_args(); dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); m=json.loads((a.cache/'manifest.json').read_text(encoding='utf-8')); net=nn.Sequential(nn.Linear(14,64),nn.ReLU(),nn.Linear(64,32),nn.ReLU(),nn.Linear(32,1)).to(dev); net.load_state_dict(torch.load(a.model,map_location='cpu')['gate']); net.eval(); temps=[i/100 for i in range(50,201)]; sums={t:0.0 for t in temps}; count=0; start=time.time(); val=[x for x in m['items'] if x['split']=='val']; test=[x for x in m['items'] if x['split']=='test']
 with torch.no_grad():
  for i,it in enumerate(val,1):
   d=torch.load(a.cache/it['file'],map_location='cpu'); f=d['features'].float(); y=d['labels'].float()
   for st in range(0,len(y),a.batch_size):
    z=net(augment(f[st:st+a.batch_size]).to(dev)).squeeze(2); yy=y[st:st+a.batch_size].to(dev)
    for t in temps: sums[t]+=nn.functional.binary_cross_entropy_with_logits(z/t,yy,reduction='sum').item()
    count+=yy.numel()
   print(f'val_file_progress={i}/{len(val)} elapsed={time.time()-start:.1f}s',flush=True)
 best=min(temps,key=lambda t:sums[t]/count); before=after=0.0; test_count=0
 with torch.no_grad():
  for i,it in enumerate(test,1):
   d=torch.load(a.cache/it['file'],map_location='cpu'); f=d['features'].float(); y=d['labels'].float()
   for st in range(0,len(y),a.batch_size):
    z=net(augment(f[st:st+a.batch_size]).to(dev)).squeeze(2); yy=y[st:st+a.batch_size].to(dev); before+=nn.functional.binary_cross_entropy_with_logits(z,yy,reduction='sum').item(); after+=nn.functional.binary_cross_entropy_with_logits(z/best,yy,reduction='sum').item(); test_count+=yy.numel()
   print(f'test_file_progress={i}/{len(test)} elapsed={time.time()-start:.1f}s',flush=True)
 print(f'best_temperature={best:.2f} val_bce={sums[best]/count:.6f} test_bce_before={before/test_count:.6f} test_bce_after={after/test_count:.6f}',flush=True); a.save.parent.mkdir(parents=True,exist_ok=True); a.save.write_text(json.dumps({'temperature':best},indent=2),encoding='utf-8'); print(f'calibration_saved={a.save}',flush=True)
if __name__=='__main__': main()
