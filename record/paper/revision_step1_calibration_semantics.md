# Revision Step 1: Calibration and Score-Mass Semantics

## Problem

The previous manuscript used "calibrated probability" for both the sigmoid
output evaluated by BCE/Brier/ECE and the softmax-normalized quantity used by
dynamic candidate selection. These quantities have different meanings for
multi-label Top-K routing.

## Locked definitions

1. Marginal expert-inclusion probability:

   \[
   q_{t,\ell,e}=\sigma(s_{t,\ell,e}/T_g^*).
   \]

   The model-specific temperature is fitted on validation multi-hot labels by
   binary cross-entropy. BCE, Brier score, and ECE evaluate this quantity.

2. Temperature-scaled ranking weight:

   \[
   w_{t,\ell,e}=\frac{\exp(s_{t,\ell,e}/T_g^*)}
   {\sum_j\exp(s_{t,\ell,j}/T_g^*)}.
   \]

   The weights sum to one and summarize relative score concentration. For
   Top-K routing with K greater than one, they are not interpreted as
   categorical probabilities.

3. Cumulative score mass:

   \[
   K_{t,\ell}(m)=\min\left\{k:\sum_{i=1}^{k}w_{t,\ell,(i)}\ge m\right\}.
   \]

   The threshold m is an operating parameter selected by validation Recall and
   average candidate width. It is not a nominal probability-coverage target.

4. Cache admission value:

   Cache rejection and replacement compare temperature-scaled ranking weights,
   not calibrated marginal inclusion probabilities.

## Consequence for existing experiments

This revision changes terminology and mathematical interpretation only. It
does not change the temperature, expert ranking, selected candidate sets,
reported Recall/Precision, or cache-replay results. No retraining is required.

## Files updated

- `F:\MOEresearch\code\evaluate_adaptive_cache_v3.py`
- `F:\MOEresearch\code\sweep_dynamic_topk_v3.py`
- `F:\MOEresearch\code\routecast_rc\model.py`
- `F:\MOEresearch\record\paper\routecast_v2_consistent_pre_experimental_design.tex`
