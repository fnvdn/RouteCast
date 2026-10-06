"""Reproduce and adapt the conditional-heatmap predictor from Patterns behind Chaos.

Two tasks are evaluated with request-disjoint train/validation/test splits:
  cross_token: current token, same layer -> next decode token, same layer
  cross_layer: current token, current MoE layer -> same token, next MoE layer

The paper selects candidates from conditional-probability heatmap rows.  For a
fair fixed-budget comparison, rows belonging to all currently active experts
are averaged and the global Top-K is returned.  No validation/test transition
is used to build the heatmaps or popularity backoff.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch

from routecast_rc.registry import ModelSpec, default_specs, _route


BUDGETS = {
    "qwen3": (8, 12, 16),
    "deepseek_r1": (8, 12, 16),
    "llama4_maverick": (1, 2, 4, 8),
}


def request_split(path: Path, root: Path) -> str:
    key = str(path.relative_to(root)).replace("\\", "/")
    value = int.from_bytes(hashlib.sha1(key.encode()).digest()[:4], "big") / 2**32
    return "train" if value < 0.70 else ("val" if value < 0.85 else "test")


def load_request(path: Path, layers: tuple[int, ...]):
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not raw:
        return [], []
    first = raw[0]
    lengths = [len(v) for v in first.values() if isinstance(v, list)]
    prefill_len = max(lengths, default=0)
    prefill = []
    for token in range(prefill_len):
        row = {}
        for layer in layers:
            value = first.get(str(layer))
            row[layer] = [int(x) for x in value[token]] if value and token < len(value) else None
        prefill.append(row)
    decode = [
        {layer: _route(record.get(str(layer)), "last") for layer in layers}
        for record in raw[1:]
    ]
    return prefill, decode


class HeatmapPredictor:
    def __init__(self, spec: ModelSpec):
        self.spec = spec
        self.counts = defaultdict(lambda: defaultdict(Counter))
        self.popularity = defaultdict(Counter)
        self.matrices: dict[tuple, torch.Tensor] = {}
        self.backoff: dict[tuple, torch.Tensor] = {}

    def add(self, task: str, key: tuple, current: list[int], target: list[int]):
        full_key = (task,) + key
        self.popularity[full_key].update(target)
        for expert in current:
            self.counts[full_key][expert].update(target)

    def finalize(self):
        e = self.spec.num_experts
        for key in set(self.counts) | set(self.popularity):
            pop = torch.ones(e, dtype=torch.float32) * 1e-8
            for expert, count in self.popularity[key].items():
                pop[expert] += count
            pop /= pop.sum()
            matrix = pop.repeat(e, 1)
            for source, counter in self.counts[key].items():
                row = torch.ones(e, dtype=torch.float32) * 1e-8
                for target, count in counter.items():
                    row[target] += count
                matrix[source] = row / row.sum()
            self.matrices[key] = matrix
            self.backoff[key] = pop

    def scores(self, task: str, key: tuple, current: list[int]) -> torch.Tensor:
        full_key = (task,) + key
        matrix = self.matrices.get(full_key)
        if matrix is None:
            return torch.full((self.spec.num_experts,), 1 / self.spec.num_experts)
        valid = [x for x in current if 0 <= x < self.spec.num_experts]
        return matrix[valid].mean(0) if valid else self.backoff[full_key]


def train_predictor(spec: ModelSpec, paths: list[Path]) -> HeatmapPredictor:
    predictor = HeatmapPredictor(spec)
    layer_pairs = list(zip(spec.active_layers[:-1], spec.active_layers[1:]))
    started = time.time()
    for index, path in enumerate(paths, 1):
        prefill, decode = load_request(path, spec.active_layers)
        timeline = prefill + decode
        # Exact paper task: cross-token conditional heatmap, one per layer.
        for left, right in zip(timeline[:-1], timeline[1:]):
            for layer in spec.active_layers:
                current, target = left.get(layer), right.get(layer)
                if current is not None and target is not None:
                    predictor.add("cross_token", (layer,), current, target)
        # Adaptation to our target: adjacent active MoE layers in the same token.
        for row in timeline:
            for source_layer, target_layer in layer_pairs:
                current, target = row.get(source_layer), row.get(target_layer)
                if current is not None and target is not None:
                    predictor.add(
                        "cross_layer", (source_layer, target_layer), current, target
                    )
        if index == 1 or index % 25 == 0 or index == len(paths):
            print(
                f"{spec.name} train_heatmap={index}/{len(paths)} "
                f"elapsed={time.time()-started:.1f}s",
                flush=True,
            )
    predictor.finalize()
    return predictor


def empty_metrics(budgets):
    return {
        k: {"hits": 0, "truth": 0, "predicted": 0, "examples": 0, "rr": 0.0, "ndcg": 0.0}
        for k in budgets
    }


def update(metrics, scores: torch.Tensor, target: list[int], budgets):
    truth = set(target)
    order = scores.argsort(descending=True).tolist()
    target_ranks = sorted(i + 1 for i, expert in enumerate(order) if expert in truth)
    for k in budgets:
        predicted = set(order[: min(k, len(order))])
        hit = len(predicted & truth)
        ideal = sum(1 / math.log2(i + 2) for i in range(min(len(truth), k)))
        dcg = sum(1 / math.log2(rank + 1) for rank in target_ranks if rank <= k)
        row = metrics[k]
        row["hits"] += hit
        row["truth"] += len(truth)
        row["predicted"] += min(k, len(order))
        row["examples"] += 1
        row["rr"] += 1 / target_ranks[0] if target_ranks else 0
        row["ndcg"] += dcg / ideal if ideal else 0


def finalize_metrics(result):
    output = {}
    for k, row in result.items():
        output[str(k)] = {
            "examples": row["examples"],
            "hits": row["hits"],
            "truth": row["truth"],
            "predicted": row["predicted"],
            "recall": row["hits"] / max(row["truth"], 1),
            "precision": row["hits"] / max(row["predicted"], 1),
            "avg_hits": row["hits"] / max(row["examples"], 1),
            "waste": 1 - row["hits"] / max(row["predicted"], 1),
            "mrr": row["rr"] / max(row["examples"], 1),
            "ndcg": row["ndcg"] / max(row["examples"], 1),
        }
    return output

def evaluate_task(spec, predictor, paths, task, budgets, progress_every):
    result = empty_metrics(budgets)
    request_rows = []
    pairs = list(zip(spec.active_layers[:-1], spec.active_layers[1:]))
    started = time.time()
    for index, path in enumerate(paths, 1):
        local = empty_metrics(budgets)
        prefill, decode = load_request(path, spec.active_layers)
        if task == "cross_token":
            # The target token must be in decode. Include last-prefill -> first-decode.
            context = ([prefill[-1]] if prefill else []) + decode
            for left, right in zip(context[:-1], context[1:]):
                for layer in spec.active_layers:
                    current, target = left.get(layer), right.get(layer)
                    if current is not None and target is not None:
                        scores=predictor.scores(task, (layer,), current)
                        update(result, scores, target, budgets); update(local, scores, target, budgets)
        else:
            for row in decode:
                for source_layer, target_layer in pairs:
                    current, target = row.get(source_layer), row.get(target_layer)
                    if current is not None and target is not None:
                        scores=predictor.scores(task, (source_layer, target_layer), current)
                        update(result, scores, target, budgets); update(local, scores, target, budgets)
        local_final=finalize_metrics(local)
        request_key=str(path.relative_to(Path(spec.root))).replace('\\','/')
        for k,values in local_final.items(): request_rows.append({'request_file':request_key,'task':task,'k':int(k),**values})
        if index == 1 or index % progress_every == 0 or index == len(paths):
            print(
                f"{spec.name} {task} test_progress={index}/{len(paths)} "
                f"elapsed={time.time()-started:.1f}s",
                flush=True,
            )
    return finalize_metrics(result),request_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--limit-per-model", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--progress-every", type=int, default=20)
    args = parser.parse_args()

    report = {
        "method": "Patterns-behind-Chaos conditional heatmap",
        "split": "request-level SHA1 70/15/15",
        "limit_per_model": args.limit_per_model,
        "fixed_budget_rule": "mean conditional rows then global Top-K",
        "models": {},
    }
    csv_rows = []
    request_rows = []
    for spec in default_specs(args.data_root):
        root = Path(spec.root)
        paths = sorted(p for p in root.rglob("*.json") if ".cache" not in p.parts)
        if args.limit_per_model > 0:
            paths = paths[: args.limit_per_model]
        grouped = {name: [] for name in ("train", "val", "test")}
        for path in paths:
            grouped[request_split(path, root)].append(path)
        print(
            f"model={spec.name} requests={len(paths)} train={len(grouped['train'])} "
            f"val={len(grouped['val'])} test={len(grouped['test'])} "
            f"experts={spec.num_experts} native_k={spec.native_topk}",
            flush=True,
        )
        predictor = train_predictor(spec, grouped["train"])
        model_report = {
            "num_experts": spec.num_experts,
            "native_topk": spec.native_topk,
            "requests": {key: len(value) for key, value in grouped.items()},
        }
        for task in ("cross_token", "cross_layer"):
            metrics,task_request_rows = evaluate_task(
                spec,
                predictor,
                grouped["test"],
                task,
                BUDGETS[spec.name],
                args.progress_every,
            )
            for row in task_request_rows: request_rows.append({'model':spec.name,**row})
            model_report[task] = metrics
            for k, values in metrics.items():
                csv_rows.append({"model": spec.name, "task": task, "k": k, **values})
                print(
                    f"{spec.name:18s} {task:11s} K={int(k):2d} "
                    f"recall={values['recall']:.6f} precision={values['precision']:.6f} "
                    f"avg_hits={values['avg_hits']:.3f} waste={values['waste']:.6f} "
                    f"mrr={values['mrr']:.6f} ndcg={values['ndcg']:.6f}",
                    flush=True,
                )
        report["models"][spec.name] = model_report

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    request_csv=args.output.with_name(args.output.stem+'_request_level.csv')
    with request_csv.open('w',newline='',encoding='utf-8-sig') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(request_rows[0])); writer.writeheader(); writer.writerows(request_rows)
    print("=" * 72, flush=True)
    print("PAPER HEATMAP SUMMARY", flush=True)
    for model, values in report["models"].items():
        native = str(values["native_topk"])
        token = values["cross_token"][native]["recall"]
        layer = values["cross_layer"][native]["recall"]
        print(
            f"{model:18s} native_K={native:>2s} "
            f"paper_cross_token_recall={token:.6f} "
            f"adapted_cross_layer_recall={layer:.6f}",
            flush=True,
        )
    print(f"json_saved={args.output}", flush=True)
    print(f"csv_saved={csv_path}", flush=True)
    print(f"request_metrics_saved={request_csv}", flush=True)


if __name__ == "__main__":
    main()
