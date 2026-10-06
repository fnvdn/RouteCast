# Revision Step 4: Naming and causal prediction timeline

## Changes completed

- Replaced the visible experimental label `RouteCast V3` with `RouteCast` in the main baseline figure. Version suffixes remain only in checkpoint and report filenames for provenance.
- Added an explicit causal timeline to Fig. 1:
  observed routes through token `t` -> forecast token `t+1` -> select/admit candidates -> speculative transfer window -> native route reveals demand -> execute from cache or demand-load.
- Kept the native router visually and textually authoritative: RouteCast proposes transfers but never changes expert selection or model output.
- Marked transfer completion in the timeline as idealized, matching the stated boundary of trace-driven cache replay.
- Updated terminology in Fig. 1 to distinguish marginal calibration from unit-sum ranking weights and score-mass budgeting.

## Submission artifacts

- `figures_final/Fig1_Framework.pdf`
- `figures_final/Fig2a_MainComparison.pdf`

## Reviewer-facing purpose

These changes remove ambiguity about when information becomes available, what RouteCast predicts, when a speculative transfer can occur, and which component remains responsible for the final expert decision.
