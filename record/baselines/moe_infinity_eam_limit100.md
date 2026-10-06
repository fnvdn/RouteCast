# MoE-Infinity EAM trace-level reproduction

## Prediction quality

| Model | K | Recall | Precision | Best neighbors |
|---|---:|---:|---:|---:|
| qwen3 | 8 | 0.447791 | 0.447791 | 5 |
| qwen3 | 12 | 0.556996 | 0.371331 | 5 |
| qwen3 | 16 | 0.637341 | 0.318670 | 5 |
| deepseek_r1 | 8 | 0.145888 | 0.145888 | 5 |
| deepseek_r1 | 12 | 0.186521 | 0.124347 | 5 |
| deepseek_r1 | 16 | 0.220717 | 0.110359 | 5 |
| llama4_maverick | 1 | 0.244873 | 0.244873 | 10 |
| llama4_maverick | 2 | 0.353977 | 0.176988 | 10 |
| llama4_maverick | 4 | 0.471653 | 0.117913 | 10 |
| llama4_maverick | 8 | 0.596246 | 0.074531 | 10 |

## Cache replay

| Model | Capacity | Method | Hit rate | Total transfers | Prefetch precision | Wasted prefetches |
|---|---:|---|---:|---:|---:|---:|
| qwen3 | 1x | LRU | 0.375097 | 1017392 | 0.000000 | 0 |
| qwen3 | 1x | MoE-Infinity EAM | 0.439710 | 1122624 | 0.090758 | 191329 |
| qwen3 | 1x | RouteCast | 0.439321 | 1730925 | 0.233996 | 626663 |
| qwen3 | 2x | LRU | 0.657425 | 557739 | 0.000000 | 0 |
| qwen3 | 2x | MoE-Infinity EAM | 0.628839 | 615060 | 0.944903 | 594 |
| qwen3 | 2x | RouteCast | 0.604360 | 1213452 | 0.457519 | 308844 |
| qwen3 | 4x | LRU | 0.844794 | 252687 | 0.000000 | 0 |
| qwen3 | 4x | MoE-Infinity EAM | 0.831697 | 284061 | 0.945174 | 551 |
| qwen3 | 4x | RouteCast | 0.801507 | 489138 | 0.804448 | 32457 |
| deepseek_r1 | 1x | LRU | 0.174686 | 822046 | 0.000000 | 0 |
| deepseek_r1 | 1x | MoE-Infinity EAM | 0.137412 | 1098340 | 0.055388 | 225921 |
| deepseek_r1 | 1x | RouteCast | 0.199576 | 1534278 | 0.106794 | 658314 |
| deepseek_r1 | 2x | LRU | 0.330946 | 666405 | 0.000000 | 0 |
| deepseek_r1 | 2x | MoE-Infinity EAM | 0.214176 | 805941 | 0.512075 | 11334 |
| deepseek_r1 | 2x | RouteCast | 0.273045 | 1365771 | 0.175973 | 528774 |
| deepseek_r1 | 4x | LRU | 0.450116 | 547706 | 0.000000 | 0 |
| deepseek_r1 | 4x | MoE-Infinity EAM | 0.332413 | 675824 | 0.890635 | 1190 |
| deepseek_r1 | 4x | RouteCast | 0.374525 | 1124343 | 0.302700 | 349588 |
| llama4_maverick | 1x | LRU | 0.284207 | 26387 | 0.000000 | 0 |
| llama4_maverick | 1x | MoE-Infinity EAM | 0.250326 | 39563 | 0.027920 | 11594 |
| llama4_maverick | 1x | RouteCast | 0.215522 | 56323 | 0.106225 | 24493 |
| llama4_maverick | 2x | LRU | 0.359565 | 23609 | 0.000000 | 0 |
| llama4_maverick | 2x | MoE-Infinity EAM | 0.364339 | 24680 | 0.172414 | 1032 |
| llama4_maverick | 2x | RouteCast | 0.308811 | 48988 | 0.175515 | 19382 |
| llama4_maverick | 4x | LRU | 0.480740 | 19142 | 0.000000 | 0 |
| llama4_maverick | 4x | MoE-Infinity EAM | 0.497559 | 18777 | 0.850980 | 38 |
| llama4_maverick | 4x | RouteCast | 0.420030 | 39976 | 0.285868 | 13280 |

> Scope: trace-level reproduction under an identical expert-entry cache model. This is not an end-to-end latency or throughput reproduction of the original offloading system.
