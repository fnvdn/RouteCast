# Revision Step 3: Idealized and Deadline-Aware Cache Replay

## Current reported experiment

The cache results already reported in the manuscript are now defined as an
**idealized cache-event replay**. The event order is:

1. construct next-token candidates from observed routes;
2. apply score-mass selection and low-weight rejection;
3. remove resident candidates and apply value-based admission;
4. assume every admitted speculative transfer completes;
5. reveal the native route as demand;
6. record hit, on-demand load, useful prefetch, wasted prefetch, and eviction.

This replay measures residency and transfer counts. It does not model expert
size, bandwidth, transfer deadline, computation overlap, kernel scheduling, or
contention. It is therefore an upper bound on the residency benefit of issued
prefetches and cannot establish latency or throughput gains.

## Optional deadline-aware sensitivity

Let S be expert size in bytes, B the effective transfer bandwidth in bytes per
second, Delta the prediction lead-time window in seconds, and L0 the fixed
startup latency. The number of complete expert transfers available in one
token window is

\[
b = \left\lfloor
\frac{B\max(0,\Delta-L_0)}{S}
\right\rfloor.
\]

Candidates are ordered by temperature-scaled ranking weight. Only the first b
admitted non-resident candidates can complete before demand. Remaining
candidates are counted as deadline-rejected and are not inserted into cache.

The implementation accepts `--deadline-transfer-budgets`, for example:

```text
--deadline-transfer-budgets 0 1 2 4 8
```

This produces sensitivity policies named `adaptive_deadline_b0`,
`adaptive_deadline_b1`, and so forth. A separate helper converts a measured or
explicitly parameterized hardware profile into b. Profiles must be labelled
as measured or parameterized; hypothetical values must not be presented as
hardware evidence.

## Files updated

- `F:\MOEresearch\code\evaluate_adaptive_cache_v3.py`
- `F:\MOEresearch\code\compute_deadline_transfer_budget.py`
- `F:\MOEresearch\record\paper\routecast_v2_consistent_pre_experimental_design.tex`
