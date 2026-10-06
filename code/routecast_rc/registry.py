from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json


@dataclass(frozen=True)
class ModelSpec:
    name: str
    root: str
    num_experts: int
    native_topk: int
    active_layers: tuple[int, ...]

    def to_dict(self):
        d = asdict(self)
        d["active_layers"] = list(self.active_layers)
        return d


def _route(value, prefill_policy: str = "last") -> list[int] | None:
    if value is None or value == []:
        return None
    if isinstance(value[0], list):
        value = value[-1] if prefill_policy == "last" else value[0]
    return [int(x) for x in value]


def inspect_trace_root(name: str, root: str | Path, sample_files: int = 8) -> ModelSpec:
    root = Path(root)
    paths = sorted(p for p in root.rglob("*.json") if ".cache" not in p.parts)
    if not paths:
        raise FileNotFoundError(f"no JSON traces under {root}")
    layers, widths = set(), set()
    max_expert = -1
    for path in paths[:sample_files]:
        records = json.loads(path.read_text(encoding="utf-8"))
        for record in records[: min(len(records), 16)]:
            for key, value in record.items():
                route = _route(value)
                if route is None:
                    continue
                layers.add(int(key)); widths.add(len(route))
                if route:
                    max_expert = max(max_expert, max(route))
    if len(widths) != 1:
        raise ValueError(f"inconsistent Top-K widths in {root}: {sorted(widths)}")
    return ModelSpec(name, str(root), max_expert + 1, next(iter(widths)), tuple(sorted(layers)))


def default_specs(data_root: str | Path) -> list[ModelSpec]:
    data_root = Path(data_root)
    roots = {
        "qwen3": data_root / "moe_trace/qwen3_complete/Qwen/Qwen3-235B-A22B-FP8",
        "deepseek_r1": data_root / "cognitivecomputations/DeepSeek-R1-AWQ",
        "llama4_maverick": data_root / "meta-llama/Llama-4-Maverick-17B-128E-Instruct",
    }
    return [inspect_trace_root(name, root) for name, root in roots.items()]

