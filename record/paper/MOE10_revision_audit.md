# MOE10 revision audit

Date: 2026-08-26

## Completed revisions

- Standardized model names as Qwen3, DeepSeek-R1, and Llama 4 Maverick.
- Standardized experiment terminology: primary evaluation (1,000 requests per model), development-scale evaluation (100 requests per model), and native-budget Recall.
- Updated the abstract, Introduction contributions, Results, Discussion, and Conclusion with the cross-model ablation result.
- Reports that the shared model is within 0.0007 macro-average Recall of three independently trained models.
- Does not attribute an accuracy gain to Group-DRO; the measured macro-Recall change is below 0.0001.
- Compressed duplicated numerical discussion in Results and Discussion.
- Reduced displayed equation environments from 27 to 10 by retaining only equations required to reproduce the method and moving secondary definitions into prose.
- Inserted float barriers at method and Results boundaries to prevent method figures and evaluation figures from accumulating on the same pages.
- Verified that all LaTeX labels referenced in the manuscript exist.
- Verified that no fixed textual figure numbers such as `Fig. 2` or `Figure 3` remain.
- Verified that all ten referenced graphics exist in `figures_final`.
- Verified balanced braces and matched LaTeX environments.
- Verified that all 20 bibliography entries are cited and that no citation key is missing.
- Corrected Fiddler to the official ICLR 2025 record and added the omitted author Tian Tang.
- Retained MoE-Infinity and Patterns behind Chaos as arXiv references because their checked official arXiv records do not establish a formal proceedings citation.

## Precision policy

- Main headline Recall values use four decimals.
- Cross-model ablation values use six decimals because the measured differences are smaller than 0.001.

## Remaining Overleaf-only checks

- Compile the revised source with the exact IEEE template used for submission.
- Inspect placement of the three method figures and confirm that no float-only page remains.
- Confirm table widths at the final paper size.
- Confirm page breaks after the inserted `\FloatBarrier` commands.
- Check the final generated figure numbering and reference pagination.
- Re-run the bibliography checker if the target venue requires DOI or URL fields in a different house style.

## MOE11 layout correction

- Whole-PDF rendering identified a nearly empty right column on page 7 and a
  large empty lower region on page 11.
- The cause was an over-constrained float schedule: five intermediate
  `\FloatBarrier` commands forced pending two-column figures to the next page.
- Removed intermediate barriers inside the Method and Results narratives; kept
  only the barrier before Discussion, where a genuine semantic boundary is
  required.
- Added bottom-float allowance and explicit top alignment for double-column
  float pages.
- Reduced the float-only page threshold from 0.80 to 0.70 so a legitimate
  double-column figure can be placed without stranding a text column.
- Source-level syntax remains balanced. Final page placement requires one
  Overleaf compile because no local TeX engine is installed in this workspace.
