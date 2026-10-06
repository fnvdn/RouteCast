"""Build the main comparison among heatmap, request-aware, GRU-only and RouteCast."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--paper',type=Path,required=True); ap.add_argument('--request-aware',type=Path,required=True)
    ap.add_argument('--gru',type=Path,required=True); ap.add_argument('--routecast',type=Path,required=True); ap.add_argument('--output-dir',type=Path,required=True); a=ap.parse_args()
    paper=json.loads(a.paper.read_text(encoding='utf-8')); req=json.loads(a.request_aware.read_text(encoding='utf-8')); gru=json.loads(a.gru.read_text(encoding='utf-8')); route=json.loads(a.routecast.read_text(encoding='utf-8'))
    rows=[]
    for model,spec in route['test'].items():
        k=str(spec['native_topk']); values={
            'Paper Heatmap':paper['models'][model]['cross_token'][k]['recall'],
            'Request-aware Heatmap':req['models'][model]['test'][k]['recall'],
            'GRU-only':gru['test'][model]['metrics'][k]['recall'],
            'RouteCast V3':route['test'][model]['metrics'][k]['recall'],
        }; full=values['RouteCast V3']
        for method,value in values.items(): rows.append({'model':model,'native_k':int(k),'method':method,'recall':value,'delta_vs_routecast':value-full})
    a.output_dir.mkdir(parents=True,exist_ok=True); csvp=a.output_dir/'strong_baselines_limit100.csv'; jsonp=a.output_dir/'strong_baselines_limit100.json'; mdp=a.output_dir/'strong_baselines_limit100.md'
    with csvp.open('w',newline='',encoding='utf-8-sig') as f: w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    jsonp.write_text(json.dumps({'rows':rows},indent=2),encoding='utf-8')
    lines=['# Cross-token 强基线比较（limit=100）','','| 模型 | 原生K | 方法 | Recall | 相对RouteCast |','|---|---:|---|---:|---:|']
    for r in rows: lines.append(f"| {r['model']} | {r['native_k']} | {r['method']} | {r['recall']:.6f} | {r['delta_vs_routecast']:+.6f} |")
    mdp.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('='*72); print('STRONG BASELINE SUMMARY')
    for r in rows: print(f"{r['model']:18s} {r['method']:24s} K={r['native_k']:2d} recall={r['recall']:.6f} delta_vs_routecast={r['delta_vs_routecast']:+.6f}")
    print(f'json_saved={jsonp}'); print(f'csv_saved={csvp}'); print(f'md_saved={mdp}')
if __name__=='__main__': main()
