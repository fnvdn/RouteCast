"""Convert a parameterized transfer profile into a per-token transfer budget."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def completed_transfers(expert_size_mib: float, bandwidth_gbps: float,
                        lead_time_ms: float, startup_latency_us: float) -> int:
    if expert_size_mib <= 0 or bandwidth_gbps <= 0:
        raise ValueError("expert size and bandwidth must be positive")
    if lead_time_ms < 0 or startup_latency_us < 0:
        raise ValueError("lead time and startup latency must be non-negative")
    usable_seconds = max(0.0, lead_time_ms / 1_000 - startup_latency_us / 1_000_000)
    transferable_bytes = bandwidth_gbps * 1_000_000_000 * usable_seconds
    expert_bytes = expert_size_mib * 1024 * 1024
    return math.floor(transferable_bytes / expert_bytes)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expert-size-mib", type=float, required=True)
    parser.add_argument("--effective-bandwidth-gbps", type=float, required=True)
    parser.add_argument("--lead-time-ms", type=float, required=True)
    parser.add_argument("--startup-latency-us", type=float, default=0.0)
    parser.add_argument("--name", default="parameterized_profile")
    parser.add_argument("--save", type=Path)
    args = parser.parse_args()
    budget = completed_transfers(
        args.expert_size_mib,
        args.effective_bandwidth_gbps,
        args.lead_time_ms,
        args.startup_latency_us,
    )
    result = {
        "name": args.name,
        "profile_type": "parameterized_not_measured",
        "expert_size_mib": args.expert_size_mib,
        "effective_bandwidth_gbps": args.effective_bandwidth_gbps,
        "lead_time_ms": args.lead_time_ms,
        "startup_latency_us": args.startup_latency_us,
        "completed_expert_transfers_per_token_window": budget,
    }
    print(json.dumps(result, indent=2), flush=True)
    print(f"replay_argument=--deadline-transfer-budgets {budget}", flush=True)
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        args.save.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"profile_saved={args.save}", flush=True)


if __name__ == "__main__":
    main()
