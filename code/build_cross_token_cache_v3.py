"""Build RouteCast V3 caches for next-token, same-layer expert forecasting."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch

from routecast_rc.registry import default_specs, _route


def split(path: Path, root: Path) -> str:
    key = str(path.relative_to(root)).replace("\\", "/")
    value = int.from_bytes(hashlib.sha1(key.encode()).digest()[:4], "big") / 2**32
    return "train" if value < .70 else ("val" if value < .85 else "test")


def load_request(path, layers):
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not raw:
        return [], []
    first = raw[0]
    n = max((len(v) for v in first.values() if isinstance(v, list)), default=0)
    prefill = []
    for token in range(n):
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


def entropy(prob):
    q = prob[prob > 0].float()
    if not len(q):
        return 0.0
    return float(-(q * q.log()).sum() / torch.log(torch.tensor(float(len(prob)))))


def make_matrices(source, experts):
    output = {}
    for key, rows in source.items():
        matrix = torch.zeros(experts, experts)
        for current, counter in rows.items():
            total = sum(counter.values()) or 1
            for target, count in counter.items():
                matrix[current, target] = count / total
        output[key] = matrix
    return output


def pack_binary(tensor: torch.Tensor) -> torch.Tensor:
    """Losslessly pack the final 0/1 dimension into uint8 bits."""
    width = tensor.shape[-1]
    padding = (-width) % 8
    if padding:
        tensor = torch.nn.functional.pad(tensor, (0, padding))
    shape = (*tensor.shape[:-1], tensor.shape[-1] // 8, 8)
    bits = tensor.to(torch.uint8).reshape(shape)
    weights = (1 << torch.arange(8, dtype=torch.uint8)).view(*([1] * (bits.ndim - 1)), 8)
    return (bits * weights).sum(-1).to(torch.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit-per-model", type=int, default=100)
    parser.add_argument("--history", type=int, default=16)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--resume", action="store_true", help="Reuse valid cache shards already present")
    parser.add_argument("--no-pack", action="store_true", help="Disable lossless bit packing (uses much more disk)")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    specs = default_specs(args.data_root)
    manifest = {
        "version": 3,
        "task": "cross_token",
        "target": "current token same layer -> next token same layer",
        "history": args.history,
        "binary_packed": not args.no_pack,
        "models": {},
        "items": [],
    }
    print("detected_models=" + json.dumps([s.to_dict() for s in specs]), flush=True)

    for spec in specs:
        root = Path(spec.root)
        paths = sorted(p for p in root.rglob("*.json") if ".cache" not in p.parts)
        if args.limit_per_model > 0:
            paths = paths[:args.limit_per_model]
        training = [p for p in paths if split(p, root) == "train"]
        popularity = defaultdict(Counter)
        one_step = defaultdict(lambda: defaultdict(Counter))
        two_step = defaultdict(lambda: defaultdict(Counter))
        started = time.time()

        # Statistics use training requests only. Prefill and decode transitions
        # are both legal history, matching the paper heatmap baseline.
        for index, path in enumerate(training, 1):
            prefill, decode = load_request(path, spec.active_layers)
            timeline = prefill + decode
            for token in range(1, len(timeline)):
                for layer in spec.active_layers:
                    current = timeline[token - 1].get(layer)
                    target = timeline[token].get(layer)
                    if current is None or target is None:
                        continue
                    popularity[layer].update(target)
                    for expert in current:
                        one_step[layer][expert].update(target)
                    if token >= 2:
                        earlier = timeline[token - 2].get(layer)
                        if earlier is not None:
                            for expert in earlier:
                                two_step[layer][expert].update(target)
            if index == 1 or index % 25 == 0 or index == len(training):
                print(
                    f"{spec.name} statistics={index}/{len(training)} "
                    f"elapsed={time.time()-started:.1f}s",
                    flush=True,
                )

        max_layer = max(spec.active_layers)
        pop = torch.zeros(max_layer + 1, spec.num_experts)
        for layer, counter in popularity.items():
            total = sum(counter.values()) or 1
            for expert, count in counter.items():
                pop[layer, expert] = count / total
        one = make_matrices(one_step, spec.num_experts)
        two = make_matrices(two_step, spec.num_experts)
        model_dir = args.output / spec.name
        model_dir.mkdir(parents=True, exist_ok=True)

        for file_index, path in enumerate(paths, 1):
            name = f"{file_index-1:05d}.pt"
            cached_path = model_dir / name
            if args.resume and cached_path.exists():
                try:
                    cached = torch.load(cached_path, map_location="cpu")
                    sample_count = len(cached["labels"])
                    manifest["items"].append({
                        "model": spec.name,
                        "file": str(Path(spec.name) / name),
                        "split": split(path, root),
                        "samples": sample_count,
                        "source": str(path.relative_to(root)),
                    })
                    if file_index == 1 or file_index % args.progress_every == 0 or file_index == len(paths):
                        print(f"{spec.name} cache={file_index}/{len(paths)} reused={name} samples={sample_count}", flush=True)
                    continue
                except (OSError, RuntimeError, KeyError, EOFError):
                    print(f"{spec.name} invalid_cache_rebuild={name}", flush=True)
            prefill, decode = load_request(path, spec.active_layers)
            prefill_frequency = torch.zeros(max_layer + 1, spec.num_experts)
            for row in prefill:
                for layer, route in row.items():
                    if route is not None:
                        prefill_frequency[layer, route] += 1
            prefill_frequency /= prefill_frequency.sum(1, keepdim=True).clamp_min(1)

            # Evaluation target is decode. Last-prefill -> first-decode is kept.
            context_timeline = ([prefill[-1]] if prefill else []) + decode
            static, history, context, labels = [], [], [], []
            past_timeline = prefill[:-1] if prefill else []
            for token in range(len(context_timeline) - 1):
                current_row = context_timeline[token]
                target_row = context_timeline[token + 1]
                # History includes the current route because it is observable.
                observed = past_timeline + context_timeline[:token + 1]
                for layer in spec.active_layers:
                    current, target = current_row.get(layer), target_row.get(layer)
                    if current is None or target is None:
                        continue
                    fallback = pop[layer].clone()
                    if fallback.sum() <= 0:
                        fallback.fill_(1 / spec.num_experts)

                    def project(route, matrix):
                        if route is None or matrix is None:
                            return fallback.clone()
                        hot = torch.zeros(spec.num_experts)
                        hot[route] = 1
                        score = hot @ matrix
                        return score / score.sum() if score.sum() > 0 else fallback.clone()

                    adjacent = project(current, one.get(layer))
                    previous = observed[-2].get(layer) if len(observed) >= 2 else None
                    skip = project(previous, two.get(layer))
                    request = prefill_frequency[layer].clone()
                    request = request if request.sum() > 0 else fallback.clone()
                    routes = [row.get(layer) for row in observed[-args.history:]]
                    hist = torch.zeros(args.history, spec.num_experts)
                    offset = args.history - len(routes)
                    for pos, route in enumerate(routes):
                        if route is not None:
                            hist[offset + pos, route] = 1
                    target_hot = torch.zeros(spec.num_experts, dtype=torch.uint8)
                    target_hot[target] = 1
                    static.append(torch.stack([fallback, adjacent, skip, request], 1).half())
                    history.append(hist.to(torch.uint8))
                    labels.append(target_hot)
                    context.append(torch.tensor([
                        layer / max_layer,
                        spec.num_experts / 256,
                        spec.native_topk / spec.num_experts,
                        entropy(request),
                        token / max(len(context_timeline) - 1, 1),
                    ], dtype=torch.float16))

            if labels:
                history_tensor = torch.stack(history)
                labels_tensor = torch.stack(labels)
                payload = {
                    "static": torch.stack(static),
                    "history": pack_binary(history_tensor) if not args.no_pack else history_tensor,
                    "context": torch.stack(context),
                    "labels": pack_binary(labels_tensor) if not args.no_pack else labels_tensor,
                    "_binary_packed": not args.no_pack,
                    "_num_experts": spec.num_experts,
                }
            else:
                packed_width = (spec.num_experts + 7) // 8
                payload = {
                    "static": torch.empty(0, spec.num_experts, 4, dtype=torch.float16),
                    "history": torch.empty(0, args.history, packed_width if not args.no_pack else spec.num_experts, dtype=torch.uint8),
                    "context": torch.empty(0, 5, dtype=torch.float16),
                    "labels": torch.empty(0, packed_width if not args.no_pack else spec.num_experts, dtype=torch.uint8),
                    "_binary_packed": not args.no_pack,
                    "_num_experts": spec.num_experts,
                }
            torch.save(payload, model_dir / name)
            manifest["items"].append({
                "model": spec.name,
                "file": str(Path(spec.name) / name),
                "split": split(path, root),
                "samples": len(labels),
                "source": str(path.relative_to(root)),
            })
            if file_index == 1 or file_index % args.progress_every == 0 or file_index == len(paths):
                print(
                    f"{spec.name} cache={file_index}/{len(paths)} samples={len(labels)} "
                    f"elapsed={time.time()-started:.1f}s",
                    flush=True,
                )
        manifest["models"][spec.name] = spec.to_dict()

    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("=" * 72, flush=True)
    print("CROSS-TOKEN CACHE SUMMARY", flush=True)
    for name, spec in manifest["models"].items():
        items = [x for x in manifest["items"] if x["model"] == name]
        counts = {part: sum(x["samples"] for x in items if x["split"] == part) for part in ("train", "val", "test")}
        print(f"{name:18s} native_K={spec['native_topk']} samples={counts}", flush=True)
    print(f"cache_saved={args.output}", flush=True)


if __name__ == "__main__":
    main()
