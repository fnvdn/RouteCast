"""Fail-fast audit for a self-contained RouteCast checkout."""
from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CODE_ROOT = PROJECT_ROOT / "code"


def require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)


def main() -> None:
    import numpy as np
    import torch

    required = [
        PROJECT_ROOT / "data" / "routecast_cache" / "cross_token_limit100_h16" / "manifest.json",
        PROJECT_ROOT / "data" / "routecast_cache" / "cross_token_limit1000_h16_packed" / "manifest.json",
        CODE_ROOT / "models" / "routecast_cross_token_limit1000_v3_stable.pt",
        CODE_ROOT / "models" / "gru_only_cross_token_limit1000_stable.pt",
        CODE_ROOT / "models" / "tcn_cross_token_limit1000.pt",
        PROJECT_ROOT / "record" / "paper" / "protocol" / "final_protocol.json",
        PROJECT_ROOT / "record" / "legacy_qwen" / "history_sweep_limit100_summary.csv",
        PROJECT_ROOT / "record" / "legacy_qwen" / "layerwise_gated_limit1000.csv",
        PROJECT_ROOT / "record" / "legacy_qwen" / "predictor_overhead_limit1000.json",
    ]
    for path in required:
        require(path)

    parse_errors: list[str] = []
    absolute_dependencies: list[str] = []
    drive_pattern = re.compile(r"(?i)(?:E:\\|E:/|F:\\MOEresearch|F:/MOEresearch)")
    for path in CODE_ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".py", ".ps1", ".bat", ".cmd"}:
            continue
        text = path.read_text(encoding="utf-8-sig")
        if path.suffix.lower() == ".py":
            try:
                ast.parse(text)
            except SyntaxError as exc:
                parse_errors.append(f"{path}: {exc}")
        if path != Path(__file__).resolve() and drive_pattern.search(text):
            absolute_dependencies.append(str(path))

    if parse_errors:
        raise RuntimeError("Python parse errors:\n" + "\n".join(parse_errors))
    if absolute_dependencies:
        raise RuntimeError("Absolute project dependencies:\n" + "\n".join(absolute_dependencies))

    manifest = json.loads(required[1].read_text(encoding="utf-8-sig"))
    if not manifest.get("items"):
        raise RuntimeError("Packed cache manifest has no items")

    checkpoint = torch.load(required[2], map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise RuntimeError("RouteCast checkpoint is not a dictionary")

    print(f"project_root={PROJECT_ROOT}")
    print(f"python={sys.executable}")
    print(f"torch={torch.__version__} cuda={torch.version.cuda} available={torch.cuda.is_available()}")
    print(f"numpy={np.__version__}")
    print(f"packed_cache_items={len(manifest['items'])}")
    print(f"checkpoint_keys={sorted(checkpoint)[:12]}")
    print("reproducibility_audit=PASS")


if __name__ == "__main__":
    main()
