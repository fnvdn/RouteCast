"""Summarize RouteCast cross-token seed runs against the paper heatmap."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--baseline-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[2024, 2025, 2026])
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    baseline = json.loads(args.baseline_report.read_text(encoding="utf-8"))
    reports = {}
    for seed in args.seeds:
        path = args.models_dir / f"routecast_cross_token_limit100_v3_seed{seed}.json"
        if seed == 2026 and not path.exists():
            path = args.models_dir / "routecast_cross_token_limit100_v3.json"
        if not path.exists():
            raise FileNotFoundError(f"missing seed report: {path}")
        reports[seed] = json.loads(path.read_text(encoding="utf-8"))

    rows, summary = [], {"seeds": args.seeds, "models": {}}
    model_names = list(reports[args.seeds[0]]["test"])
    for model in model_names:
        native_k = reports[args.seeds[0]]["test"][model]["native_topk"]
        paper = baseline["models"][model]["cross_token"][str(native_k)]["recall"]
        values = []
        for seed in args.seeds:
            value = reports[seed]["test"][model]["metrics"][str(native_k)]["recall"]
            values.append(value)
            rows.append({
                "model": model,
                "native_k": native_k,
                "seed": seed,
                "paper_recall": paper,
                "routecast_recall": value,
                "delta": value - paper,
                "exceeded": value > paper,
            })
        mean = statistics.mean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        summary["models"][model] = {
            "native_k": native_k,
            "paper_recall": paper,
            "seed_recalls": dict(zip(map(str, args.seeds), values)),
            "mean_recall": mean,
            "sample_std": std,
            "mean_delta": mean - paper,
            "all_seeds_exceeded": all(v > paper for v in values),
        }

    json_path = args.output_dir / "cross_token_seed_summary.json"
    csv_path = args.output_dir / "cross_token_seed_runs.csv"
    md_path = args.output_dir / "cross_token_seed_summary.md"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Cross-token RouteCast 随机种子复现实验",
        "",
        f"随机种子：{', '.join(map(str, args.seeds))}",
        "",
        "| 模型 | 原生K | Paper Heatmap | RouteCast均值 | 标准差 | 平均提升 | 三种子均超过 |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for model, item in summary["models"].items():
        lines.append(
            f"| {model} | {item['native_k']} | {item['paper_recall']:.6f} | "
            f"{item['mean_recall']:.6f} | {item['sample_std']:.6f} | "
            f"{item['mean_delta']:+.6f} | {item['all_seeds_exceeded']} |"
        )
    lines += ["", "## 单次结果", "", "| 模型 | Seed | Recall | Delta | Exceeded |", "|---|---:|---:|---:|---|"]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['seed']} | {row['routecast_recall']:.6f} | "
            f"{row['delta']:+.6f} | {row['exceeded']} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("=" * 72, flush=True)
    print("CROSS-TOKEN SEED REPRODUCIBILITY SUMMARY", flush=True)
    for model, item in summary["models"].items():
        print(
            f"{model:18s} K={item['native_k']:2d} paper={item['paper_recall']:.6f} "
            f"mean={item['mean_recall']:.6f} std={item['sample_std']:.6f} "
            f"delta={item['mean_delta']:+.6f} all_exceeded={item['all_seeds_exceeded']}",
            flush=True,
        )
    print(f"json_saved={json_path}", flush=True)
    print(f"csv_saved={csv_path}", flush=True)
    print(f"md_saved={md_path}", flush=True)


if __name__ == "__main__":
    main()
