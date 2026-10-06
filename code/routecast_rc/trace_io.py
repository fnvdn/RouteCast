from __future__ import annotations

import json
from pathlib import Path


def iter_trace_files(root: str | Path):
    root = Path(root)
    yield from sorted(root.rglob("*.json"))


def load_route_trace(path: str | Path):
    """Load the public selected_experts JSON format.

    Returns a list of token records. Each record maps layer IDs to expert IDs;
    prefill rows with nested token arrays are retained for the caller to filter.
    """
    with Path(path).open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, list):
        raise ValueError(f"trace must be a list: {path}")
    return obj


def decode_layer_set(record, layer_id: str | int):
    value = record.get(str(layer_id), record.get(layer_id))
    if value is None:
        return None
    if value and isinstance(value[0], list):
        # Prefill rows contain multiple input-token selections. The evaluator
        # chooses one row explicitly; do not silently flatten across tokens.
        return [list(map(int, row)) for row in value]
    return list(map(int, value))
