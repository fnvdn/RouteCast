"""Create the frozen limit=1000 main forecasting comparison table."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path

NATIVE={'qwen3':8,'deepseek_r1':8,'llama4_maverick':1}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--paper',type=Path,required=True); p.add_argument('--request-aware',type=Path,required=True)
    p.add_argument('--gru',type=Path,required=True); p.add_argument('--tcn',type=Path,required=True); p.add_argument('--routecast',type=Path,required=True); p.add_argument('--output-dir',type=Path,required=True); a=p.parse_args()
    paper=json.loads(a.paper.read_text(encoding='utf-8')); req=json.loads(a.request_aware.read_text(encoding='utf-8'))
    gru=json.loads(a.gru.read_text(encoding='utf-8')); tcn=json.loads(a.tcn.read_text(encoding='utf-8')); route=json.loads(a.routecast.read_text(encoding='utf-8')); rows=[]
    for model,k in NATIVE.items():
        sources={
            'Paper Heatmap':paper['models'][model]['cross_token'][str(k)],
            'Request-aware Heatmap':req['models'][model]['test'][str(k)],
            'GRU-only':gru['models'][model]['budgets'][str(k)],
            'TCN-only':tcn['models'][model]['budgets'][str(k)],
            'RouteCast V3':route['models'][model]['budgets'][str(k)],
        }; ref=sources['RouteCast V3']['recall']
        for method,x in sources.items(): rows.append({'model':model,'native_k':k,'method':method,**{m:float(x[m]) for m in ('recall','precision','mrr','ndcg')},'delta_recall_vs_routecast':float(x['recall'])-ref})
    a.output_dir.mkdir(parents=True,exist_ok=True); csvp=a.output_dir/'main_forecasting_limit1000.csv'; jsonp=a.output_dir/'main_forecasting_limit1000.json'; mdp=a.output_dir/'main_forecasting_limit1000.md'
    with csvp.open('w',newline='',encoding='utf-8-sig') as f: w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    jsonp.write_text(json.dumps({'protocol':'request-disjoint limit=1000; native-K; seed=2026','rows':rows},indent=2),encoding='utf-8')
    lines=['# RouteCast limit=1000 主预测结果','','| 模型 | K | 方法 | Recall | Precision | MRR | NDCG | Delta Recall |','|---|---:|---|---:|---:|---:|---:|---:|']
    for r in rows: lines.append(f"| {r['model']} | {r['native_k']} | {r['method']} | {r['recall']:.6f} | {r['precision']:.6f} | {r['mrr']:.6f} | {r['ndcg']:.6f} | {r['delta_recall_vs_routecast']:+.6f} |")
    mdp.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n================ LIMIT1000 MAIN TABLE ================')
    for r in rows: print(f"{r['model']:18s} {r['method']:24s} K={r['native_k']:2d} recall={r['recall']:.6f} mrr={r['mrr']:.6f} ndcg={r['ndcg']:.6f} delta={r['delta_recall_vs_routecast']:+.6f}")
    print(f'json_saved={jsonp}\ncsv_saved={csvp}\nmd_saved={mdp}')

if __name__=='__main__': main()
