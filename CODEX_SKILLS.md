# Codex Project Skills

## GUI and research visualization coding logic

- Do not stop at "the program runs". For every GUI or research visualization change, actively inspect whether the result is readable, whether states overlap, and whether the visual encoding can mislead analysis.
- When multiple metrics or states coexist, prefer explicit separated tracks, panels, or layers over compressed color-only encodings. For example, hop-working and delay-working should be shown as two rows instead of being merged into one row.
- If a likely UX or scientific-interpretation issue is visible during implementation, report it proactively and ask whether to improve it, even if the user did not explicitly request that exact change.
- Interactive plots should provide feedback for inspection whenever practical: hover markers, tooltips, selected-state labels, and clear current-time indicators.
- For topology viewers, keep the viewer as display-only. Topology transformation, region constraints, link setup scheduling, and metric computation should be prepared outside the GUI and passed in as explicit data.
- Before calling a GUI change finished, run at least a syntax/import check and, when feasible, an offscreen screenshot or small-window smoke test.
