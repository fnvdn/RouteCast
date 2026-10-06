# RouteCast data (not stored in Git)

The experiment data is intentionally excluded from the Git repository. The
local project uses public expert-selection traces from:

`https://huggingface.co/datasets/core12345/MoE_expert_selection_trace`

## Required local layout

```text
data/
|-- moe_trace/                  downloaded public traces
`-- routecast_cache/
    |-- cross_token_limit100_h16/
    `-- cross_token_limit1000_h16_packed/
```

The full local data tree is approximately 194 GB. A teammate can either copy
the verified `data/` directory from the project owner or download the public
traces and rebuild caches with the cache-building scripts under `code/`.

At minimum, `code/verify_reproducibility.py` expects these files:

```text
data/routecast_cache/cross_token_limit100_h16/manifest.json
data/routecast_cache/cross_token_limit1000_h16_packed/manifest.json
```

Do not commit dataset shards, packed tensors, Hugging Face cache contents, or
access tokens. When transferring the data out of band, verify file counts and
checksums before deleting the source copy.

