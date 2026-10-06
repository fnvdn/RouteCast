"""Summarize RouteCast V3 cross-token module ablations."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--models-dir',type=Path,required=True); ap.add_argument('--output-dir',type=Path,required=True); a=ap.parse_args()
    files={
        'full':a.models_dir/'routecast_cross_token_limit100_v3.json',
        'no_history':a.models_dir/'routecast_cross_token_ablation_no_history.json',
        'no_two_hop':a.models_dir/'routecast_cross_token_ablation_no_two_hop.json',
        'no_prefill':a.models_dir/'routecast_cross_token_ablation_no_prefill.json',
    }
    reports={name:json.loads(path.read_text(encoding='utf-8')) for name,path in files.items()}
    rows=[]
    for model,spec in reports['full']['test'].items():
        k=str(spec['native_topk']); full=spec['metrics'][k]['recall']
        for variant,report in reports.items():
            value=report['test'][model]['metrics'][k]['recall']
            rows.append({'model':model,'native_k':int(k),'variant':variant,'recall':value,'delta_vs_full':value-full})
    a.output_dir.mkdir(parents=True,exist_ok=True)
    csvp=a.output_dir/'cross_token_ablation_limit100.csv'; jsonp=a.output_dir/'cross_token_ablation_limit100.json'; mdp=a.output_dir/'cross_token_ablation_limit100.md'
    with csvp.open('w',newline='',encoding='utf-8-sig') as f: w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    jsonp.write_text(json.dumps({'rows':rows},indent=2),encoding='utf-8')
    lines=['# Cross-token RouteCast V3 模块消融（limit=100）','','| 模型 | 原生K | 版本 | Recall | 相对完整模型变化 |','|---|---:|---|---:|---:|']
    for r in rows: lines.append(f"| {r['model']} | {r['native_k']} | {r['variant']} | {r['recall']:.6f} | {r['delta_vs_full']:+.6f} |")
    mdp.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('='*72); print('CROSS-TOKEN ABLATION SUMMARY')
    for r in rows: print(f"{r['model']:18s} {r['variant']:14s} K={r['native_k']:2d} recall={r['recall']:.6f} delta_vs_full={r['delta_vs_full']:+.6f}")
    print(f'json_saved={jsonp}'); print(f'csv_saved={csvp}'); print(f'md_saved={mdp}')
if __name__=='__main__': main()
