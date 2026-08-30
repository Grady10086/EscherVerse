# Content-specific SFT/LoRA ablation

Three non-overlapping 1,409-example subsets were matched on question format and
normalized gold answer: Intent-focused, Dynamic-only, and Random mixture. Videos
overlapping the benchmark were excluded. Each condition was trained from
Qwen3-VL-4B-Instruct with two seeds under the same LoRA budget.

Public artifacts include:

- `subsets_strict/`: frozen training subsets and hashes;
- `manifests/eval_panel_strict.json`: the 486+486 evaluation panel;
- `scripts/`: subset construction, training, evaluation-merging, and analysis;
- `results/run_metrics/`: six training summaries;
- `results/final_runs/`: six item-level adapter outputs;
- `results/analysis_final.json`: the reported difference-in-differences and
  video-clustered bootstrap analysis.

The portable training entry point is also copied to
`../../reproducibility/training/train_qwen3vl_lora.py`. Server-specific shell
launchers are intentionally excluded; use the portable command in the
top-level README for a fresh run.
