# Additional analyses and ablations

This directory contains frozen public materials for the benchmark's supporting
analyses, controls, audits, and training ablation.

| Directory | Analysis | Public material |
|-----------|----------|-----------------|
| `concept_structure/` | Correlation, PCA, and category-pair structure | Input table, script, tests, full results |
| `matched_pair/` | Same-video 50-pair comparison | Frozen panel summary and analysis scripts |
| `perception_control/` | Necessary-perception controls on 200 items | Frozen probe manifest, model summaries, scripts |
| `failure_mode/` | Quantitative failure-mode analysis of 300 errors | Rater files, adjudication records, scripts, final weighted analysis |
| `coverage_audit/` | Six-dimension coverage audit | Authoring records, two-rater judgments, adjudication, scripts, final audit |
| `sft_ablation/` | Equal-budget intent/dynamic/random LoRA ablation | Frozen subsets and panel, training script, per-seed outputs, final bootstrap analysis |

The directories intentionally exclude videos, API credentials, private human
records, internal review applications, and discarded exploratory data versions.
See each experiment's metadata and result files for sample sizes and hashes.
