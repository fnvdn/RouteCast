# RouteCast final submission audit

Date: 2026-08-26  
Source: `routecast_v2_consistent_pre_experimental_design.tex`  
Rendered draft reviewed: `D:\MOE13.pdf`

## 1. Final status

The manuscript passes the current content, numerical-consistency, LaTeX-static,
reference-linkage, and rendered-layout checks. No blocking defect was found.
The final source contains four minor copy-edit corrections made after MOE13;
one final two-pass Overleaf compile is therefore required before submission.

## 2. Terminology ledger

| Concept | Canonical form |
|---|---|
| Proposed method | RouteCast |
| Model families | Qwen3; DeepSeek-R1; Llama 4 Maverick |
| Main scale | primary 1,000-request-per-model evaluation |
| Diagnostic scale | development-scale 100-request-per-model evaluation |
| Main metric | native-budget Recall |
| Cache experiment | idealized trace-driven cache-event replay |
| Calibration output | marginal inclusion probability |
| Budget normalization | unit-sum ranking weight / cumulative score mass |

`Llama 4 M.` is retained only in narrow tables. `primary-evaluation` is retained
only as a grammatically required compound modifier.

## 3. Numerical audit

- Main Recall: Qwen3 0.5558; DeepSeek-R1 0.3172; Llama 4 Maverick 0.3602.
- Exact RouteCast-minus-TCN differences: 0.028147, 0.045478, and 0.057871.
- Reported percentage-point gains (2.81, 4.55, 5.79) are consistent after rounding.
- Shared-model macro Recall: 0.411102; per-model macro Recall: 0.411748;
  difference: 0.000646, correctly bounded as less than 0.0007.
- Shared Group-DRO macro Recall: 0.411102; shared no-DRO macro Recall:
  0.411195; absolute difference: 0.000093, correctly bounded as less than 0.0001.
- The manuscript does not claim that Group-DRO improves Recall.
- Main-table values agree with the stored limit=1000 result records.
- Bootstrap claims agree with the stored 10,000-resample request-level reports.

## 4. Mathematical and methodological audit

- The manuscript separates sigmoid marginal inclusion probabilities from
  softmax unit-sum ranking weights.
- Temperature scaling preserves ranking and is not described as an additional
  fixed-budget forecasting gain.
- Cumulative score mass is described as an operating threshold, not as a
  categorical coverage probability for Top-K routing.
- The native router remains authoritative; speculative predictions cannot alter
  model output.
- Cache replay is consistently marked as idealized and is not used to claim
  end-to-end latency, throughput, or energy improvement.
- Six forecasting branches are used consistently throughout the abstract,
  Methods, captions, and Discussion.

## 5. LaTeX and layout audit

- Balanced braces and matched LaTeX environments.
- No duplicate labels, undefined references, missing bibliography keys, or
  uncited bibliography entries.
- No hard-coded textual figure numbers or unresolved TODO/TBD placeholders.
- All referenced graphics exist in `figures_final`.
- MOE13 contains 12 letter-size pages with no float-only page, stranded heading,
  clipped figure, visibly over-wide table, or abnormal central whitespace.
- The small residual whitespace on the final reference page is normal for an
  IEEE two-column manuscript.

## 6. Reference audit

Twenty references are cited and linked to the claims they support.

- Formal venues retained for established papers, including ISCA 2024
  Pre-gated MoE and ICLR 2025 Fiddler.
- Fiddler author list includes Tian Tang and matches the official proceedings.
- Recent works (Patterns behind Chaos, ST-MoE, SpecMD, and SpecPrefetch) remain
  explicitly identified as arXiv preprints.
- GRU, Group-DRO, AdamW, temperature scaling, TCN, NDCG, and bootstrap citations
  occur at their corresponding methodological definitions.
- DOI-bearing references for Pre-gated MoE, the GRU paper, and NDCG have
  title-compatible metadata.

## 7. Minor corrections applied during final audit

1. Replaced generic `accuracy` wording with the defined Recall metric.
2. Corrected a sentence boundary before the validation-Recall definition.
3. Corrected mathematical punctuation in the total-loss equation.
4. Retained table-only model abbreviation without propagating it into prose.

## 8. Final author actions

1. Upload the revised `.tex` file to Overleaf.
2. Compile twice to refresh all cross-references and final pagination.
3. Confirm that the compilation log contains no undefined reference, citation,
   overfull box, or float warning.
4. Download the final PDF and archive it together with this audit and the exact
   source/figure directory.

