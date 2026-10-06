# Figure QA notes

- Backend: Python (matplotlib), used exclusively for drawing, previewing and export.
- Bundle: 14 figures; each exported as editable SVG, PDF, 300-dpi PNG and 600-dpi TIFF.
- Source validation: 0 failures after explicit export checks; report in `source_validation.json`.
- PDF typography: all 14 PDFs are auditable and contain no text below 5 pt; minimum observed glyph size ranges from 5.0 to 6.5 pt.
- Visual inspection: every panel and the complete contact sheet were reviewed at rendered size. Panel labels, legends, axes and annotations remain legible without detected collisions after revision.
- Statistical unit: Figure 2d uses paired request-level bootstrap, 10,000 repeats, seed 2026. Other aggregate panels show deterministic pooled trace metrics unless a seed interval is explicitly displayed.
- Integrity: no observations were removed for plotting. No synthetic experimental values were introduced. The framework schematic contains no numerical claims.
- Evidence boundary: Figures 2 and Extended Data Figure 1 use the current stable limit=1000 protocol. Figures 3–6 and Extended Data Figures 4, 6 and 8 are cross-model limit=100 development evidence. Extended Data Figures 2, 5 and 7 are explicitly labelled legacy Qwen-only evidence.
- Remaining reviewer risk: current RouteCast V3 overhead has not been re-benchmarked, and limit=1000 multi-seed/TCN evidence remains pending. These omissions are disclosed rather than filled with legacy values.

