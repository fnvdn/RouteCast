"""Audit RouteCast cache tensors for non-finite values and label integrity."""
from __future__ import annotations
import argparse,json,time
from collections import Counter,defaultdict
from pathlib import Path
import torch

CHECK_FLOAT=('static','context')

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--max-files',type=int,default=0,help='0 audits the complete manifest')
    p.add_argument('--progress-every',type=int,default=50)
    a=p.parse_args()
    manifest=json.loads((a.cache/'manifest.json').read_text(encoding='utf-8'))
    items=manifest['items'][:a.max_files or None]; started=time.time()
    bad=[]; totals=Counter(); ranges=defaultdict(lambda:[float('inf'),float('-inf')])
    for i,item in enumerate(items,1):
        d=torch.load(a.cache/item['file'],map_location='cpu')
        n=len(d['labels']); totals['files']+=1; totals['samples']+=n
        totals[f"files_{item['model']}_{item['split']}"]+=1
        totals[f"samples_{item['model']}_{item['split']}"]+=n
        for key in CHECK_FLOAT:
            x=d[key]
            finite=torch.isfinite(x)
            if not finite.all(): bad.append({'file':item['file'],'model':item['model'],'split':item['split'],'tensor':key,'nonfinite':int((~finite).sum())})
            if finite.any():
                ranges[key][0]=min(ranges[key][0],float(x[finite].min()))
                ranges[key][1]=max(ranges[key][1],float(x[finite].max()))
        for key in ('history','labels'):
            x=d[key]
            if x.dtype.is_floating_point and not torch.isfinite(x).all():
                bad.append({'file':item['file'],'model':item['model'],'split':item['split'],'tensor':key,'nonfinite':int((~torch.isfinite(x)).sum())})
        if i==1 or i%a.progress_every==0 or i==len(items):
            print(f'audit_progress={i}/{len(items)} samples={totals["samples"]} bad={len(bad)} elapsed={time.time()-started:.1f}s',flush=True)
    report={'cache':str(a.cache),'manifest_items':len(manifest['items']),'audited_items':len(items),'complete_audit':len(items)==len(manifest['items']),
            'status':'PASS' if not bad else 'FAIL','bad_tensors':bad,'counts':dict(totals),'finite_ranges':dict(ranges),'elapsed_seconds':time.time()-started}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(f'audit_status={report["status"]} files={totals["files"]} samples={totals["samples"]} bad_tensors={len(bad)}',flush=True)
    print(f'report_saved={a.output}',flush=True)
    if bad: raise SystemExit(2)

if __name__=='__main__': main()
