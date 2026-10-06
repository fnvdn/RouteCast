"""Run and summarize the per-model and no-Group-DRO RouteCast ablations.

All runs reuse the packed limit=1000 cache and the same architecture, loss,
seed, epochs, batch size, validation checkpoint rule, and test metrics as the
primary shared Group-DRO model. Subprocess output is streamed to both the
terminal and persistent log files.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


MODELS = ("qwen3", "deepseek_r1", "llama4_maverick")


def stream_run(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("\nRUN:", subprocess.list2cmdline(command), flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return_code = process.wait()
    if return_code:
        raise SystemExit(f"run failed with exit code {return_code}: {command}")


def native_recall(report: dict, model: str) -> float:
    block = report["test"][model]
    return float(block["metrics"][str(block["native_topk"])]["recall"])


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def summarize(full_path: Path, no_dro_path: Path,
              per_model_paths: dict[str, Path], output_dir: Path) -> None:
    full = load_json(full_path)
    no_dro = load_json(no_dro_path)
    per_model = {name: load_json(path) for name, path in per_model_paths.items()}
    rows = []
    for model in MODELS:
        shared = native_recall(full, model)
        nodro = native_recall(no_dro, model)
        separate = native_recall(per_model[model], model)
        rows.append({
            "model": model,
            "shared_group_dro_recall": shared,
            "shared_no_dro_recall": nodro,
            "per_model_recall": separate,
            "group_dro_delta": shared - nodro,
            "shared_training_delta": shared - separate,
        })
    macro = {
        "model": "macro_average",
        "shared_group_dro_recall": sum(r["shared_group_dro_recall"] for r in rows) / len(rows),
        "shared_no_dro_recall": sum(r["shared_no_dro_recall"] for r in rows) / len(rows),
        "per_model_recall": sum(r["per_model_recall"] for r in rows) / len(rows),
        "group_dro_delta": sum(r["group_dro_delta"] for r in rows) / len(rows),
        "shared_training_delta": sum(r["shared_training_delta"] for r in rows) / len(rows),
    }
    rows.append(macro)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "cross_model_ablation_limit1000.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "protocol": {
            "cache": "cross_token_limit1000_h16_packed",
            "seed": 2026,
            "comparison": "shared Group-DRO vs shared uniform group weights vs independent per-model training",
        },
        "rows": rows,
    }
    json_path = output_dir / "cross_model_ablation_limit1000.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_path = output_dir / "cross_model_ablation_limit1000.md"
    lines = [
        "# Cross-model training ablation (limit=1000)", "",
        "| Model | Shared + Group-DRO | Shared, no DRO | Per-model | DRO delta | Shared-training delta |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['shared_group_dro_recall']:.6f} | "
            f"{row['shared_no_dro_recall']:.6f} | {row['per_model_recall']:.6f} | "
            f"{row['group_dro_delta']:+.6f} | {row['shared_training_delta']:+.6f} |"
        )
    lines += [
        "",
        "Positive DRO delta means adaptive Group-DRO improved Recall over uniform group weighting.",
        "Positive shared-training delta means the shared model improved Recall over independent per-model training.",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n================ CROSS-MODEL ABLATION SUMMARY ================", flush=True)
    for row in rows:
        print(
            f"{row['model']:18s} full={row['shared_group_dro_recall']:.6f} "
            f"no_dro={row['shared_no_dro_recall']:.6f} "
            f"per_model={row['per_model_recall']:.6f} "
            f"dro_delta={row['group_dro_delta']:+.6f} "
            f"shared_delta={row['shared_training_delta']:+.6f}",
            flush=True,
        )
    print(f"csv_saved={csv_path}\njson_saved={json_path}\nmd_saved={md_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--full-report", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--max-train-files", type=int, default=0)
    parser.add_argument("--max-batches-per-file", type=int, default=0)
    args = parser.parse_args()

    trainer = Path(__file__).with_name("train_universal_routecast_v3.py")
    args.model_dir.mkdir(parents=True, exist_ok=True)
    common = [
        sys.executable, str(trainer), "--cache", str(args.cache),
        "--epochs", str(args.epochs), "--batch-size", str(args.batch_size),
        "--seed", str(args.seed),
    ]
    if args.max_train_files:
        common += ["--max-train-files", str(args.max_train_files)]
    if args.max_batches_per_file:
        common += ["--max-batches-per-file", str(args.max_batches_per_file)]

    no_dro_model = args.model_dir / "routecast_limit1000_shared_no_dro.pt"
    jobs = [
        ("shared_no_dro", no_dro_model,
         common + ["--disable-group-dro", "--save", str(no_dro_model)]),
    ]
    per_paths = {}
    for model in MODELS:
        path = args.model_dir / f"routecast_limit1000_per_model_{model}.pt"
        per_paths[model] = path.with_suffix(".json")
        jobs.append((f"per_model_{model}", path,
                     common + ["--models", model, "--save", str(path)]))

    for name, model_path, command in jobs:
        report_path = model_path.with_suffix(".json")
        if args.skip_existing and report_path.exists():
            print(f"skip_existing={name} report={report_path}", flush=True)
            continue
        stream_run(command, args.output_dir / "logs" / f"{name}.log")

    summarize(
        args.full_report,
        no_dro_model.with_suffix(".json"),
        per_paths,
        args.output_dir,
    )


if __name__ == "__main__":
    main()
