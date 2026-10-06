"""Dynamic Top-K evaluation for calibrated gated fusion."""
import argparse,json,time
from pathlib import Path
import torch
from torch import nn
try:
    from scripts.calibrate_gated_cached import augment
except ModuleNotFoundError:
    from calibrate_gated_cached import augment

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--cache',type=Path,required=True); ap.add_argument('--model',type=Path,required=True); ap.add_argument('--calibration',type=Path,required=True); ap.add_argument('--mass',type=float,default=.9); ap.add_argument('--max-k',type=int,default=8,help='maximum dynamic prefetch count; training is unchanged'); ap.add_argument('--batch-size',type=int,default=8192); a=ap.parse_args(); dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); m=json.loads((a.cache/'manifest.json').read_text(encoding='utf-8')); temp=json.loads(a.calibration.read_text(encoding='utf-8'))['temperature']; net=nn.Sequential(nn.Linear(14,64),nn.ReLU(),nn.Linear(64,32),nn.ReLU(),nn.Linear(32,1)).to(dev); net.load_state_dict(torch.load(a.model,map_location='cpu')['gate']); net.eval(); subset=[x for x in m['items'] if x['split']=='test']; hit=actual=predicted=fixed=examples=0; start=time.time()
 with torch.no_grad():
  for i,it in enumerate(subset,1):
   d=torch.load(a.cache/it['file'],map_location='cpu'); f=d['features'].float(); y=d['labels'];
   for st in range(0,len(y),a.batch_size):
    z=net(augment(f[st:st+a.batch_size]).to(dev)).squeeze(2)/temp; prob=torch.softmax(z,1); vals,idx=prob.sort(1,descending=True); cum=vals.cumsum(1); ks=torch.clamp((cum<a.mass).sum(1)+1,max=min(a.max_k,idx.shape[1])); fixed_idx=idx[:,:8]
    for r in range(len(y[st:st+a.batch_size])):
     k=int(ks[r]); true=set(y[st+r].nonzero().flatten().tolist()); chosen=set(idx[r,:k].tolist()); hit+=len(chosen&true); actual+=len(true); predicted+=k; fixed+=len(set(fixed_idx[r].tolist())&true); examples+=1
   print(f'test_file_progress={i}/{len(subset)} elapsed={time.time()-start:.1f}s',flush=True)
 print(f'temperature={temp} mass={a.mass} max_k={a.max_k} examples={examples} dynamic_avg_k={predicted/examples:.4f} dynamic_recall={hit/actual:.6f} dynamic_precision={hit/predicted:.6f} fixed_top8_recall={fixed/actual:.6f}',flush=True)
if __name__=='__main__': main()
