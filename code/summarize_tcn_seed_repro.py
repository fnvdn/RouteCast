"""Summarize matched RouteCast-vs-TCN development-scale seed runs."""
from __future__ import annotations
import argparse,csv,json,statistics
from pathlib import Path

MODELS={"qwen3":8,"deepseek_r1":8,"llama4_maverick":1}

def load(path): return json.loads(path.read_text(encoding="utf-8"))

def metric(report,model,k):
    block=report["test"][model]
    metrics=block.get("metrics",block.get("budgets"))
    return float(metrics[str(k)]["recall"])

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--models-dir",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    a=p.parse_args(); rows=[]
    paths={
        2024:("routecast_cross_token_limit100_v3_seed2024.json","tcn_cross_token_limit100_seed2024.json"),
        2025:("routecast_cross_token_limit100_v3_seed2025.json","tcn_cross_token_limit100_seed2025.json"),
        2026:("routecast_cross_token_limit100_v3.json","tcn_cross_token_limit100_seed2026.json"),
    }
    for seed,(rc_name,tcn_name) in paths.items():
        rc=load(a.models_dir/rc_name); tcn=load(a.models_dir/tcn_name)
        for model,k in MODELS.items():
            r=metric(rc,model,k); t=metric(tcn,model,k)
            rows.append({"seed":seed,"model":model,"native_k":k,"routecast_recall":r,"tcn_recall":t,"delta":r-t})
    summary={}
    for model in MODELS:
        selected=[r for r in rows if r["model"]==model]; deltas=[r["delta"] for r in selected]
        summary[model]={
            "native_k":MODELS[model],"seeds":len(deltas),
            "routecast_mean":statistics.mean(r["routecast_recall"] for r in selected),
            "routecast_std":statistics.stdev(r["routecast_recall"] for r in selected),
            "tcn_mean":statistics.mean(r["tcn_recall"] for r in selected),
            "tcn_std":statistics.stdev(r["tcn_recall"] for r in selected),
            "delta_mean":statistics.mean(deltas),"delta_std":statistics.stdev(deltas),
            "all_routecast_exceeded":all(x>0 for x in deltas),
        }
    a.output_dir.mkdir(parents=True,exist_ok=True)
    json_path=a.output_dir/"tcn_routecast_seed_summary.json"
    csv_path=a.output_dir/"tcn_routecast_seed_runs.csv"
    md_path=a.output_dir/"tcn_routecast_seed_summary.md"
    json_path.write_text(json.dumps({"protocol":"limit=100 development stability screening; matched seeds 2024/2025/2026","runs":rows,"summary":summary},indent=2),encoding="utf-8")
    with csv_path.open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    lines=["# RouteCast vs TCN seed reproducibility","","Development-scale limit=100 screening; main limit=1000 results remain the primary evidence.","","| Model | K | RouteCast mean ± SD | TCN mean ± SD | Delta mean ± SD | RouteCast wins all seeds |","|---|---:|---:|---:|---:|---:|"]
    for model,x in summary.items():
        lines.append(f"| {model} | {x['native_k']} | {x['routecast_mean']:.6f} ± {x['routecast_std']:.6f} | {x['tcn_mean']:.6f} ± {x['tcn_std']:.6f} | {x['delta_mean']:+.6f} ± {x['delta_std']:.6f} | {x['all_routecast_exceeded']} |")
    md_path.write_text("\n".join(lines)+"\n",encoding="utf-8")
    print("\n================ TCN SEED REPRODUCIBILITY ================")
    for model,x in summary.items(): print(f"{model:18s} K={x['native_k']:2d} routecast={x['routecast_mean']:.6f}±{x['routecast_std']:.6f} tcn={x['tcn_mean']:.6f}±{x['tcn_std']:.6f} delta={x['delta_mean']:+.6f}±{x['delta_std']:.6f} all_wins={x['all_routecast_exceeded']}")
    print(f"json_saved={json_path}\ncsv_saved={csv_path}\nmd_saved={md_path}")

if __name__=="__main__": main()
