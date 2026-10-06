from __future__ import annotations

from collections import OrderedDict


class LRUExpertCache:
    """Trace-driven cache accounting, independent of a particular GPU."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.items = OrderedDict()
        self.hits = self.misses = self.wasted_prefetch = 0

    def access(self, experts):
        experts = set(experts)
        self.hits += len(experts & self.items.keys())
        self.misses += len(experts - self.items.keys())
        for e in experts:
            self.items.pop(e, None)
            self.items[e] = None
        while len(self.items) > self.capacity:
            self.items.popitem(last=False)

    def prefetch(self, experts, next_truth=None):
        experts = set(experts)
        if next_truth is not None:
            self.wasted_prefetch += len(experts - set(next_truth))
        for e in experts:
            self.items.pop(e, None)
            self.items[e] = None
        while len(self.items) > self.capacity:
            self.items.popitem(last=False)

    def report(self):
        total = self.hits + self.misses
        return {
            "hit_rate": self.hits / max(total, 1),
            "demand_miss": self.misses,
            "wasted_prefetch": self.wasted_prefetch,
            "resident_experts": len(self.items),
        }
