# Same-video 50-pair analysis

`frozen_panel/candidate_pairs.csv` contains the disclosed post-hoc 50-pair
panel. The four panel-construction models were used during selection; the three
held-out models were evaluated only after the panel was frozen. The complete
seven-model summary is `posthoc_natural50_model_panel_20260812.json`.

The final panel is a same-video comparison, not a causal matched-pair design.
Its 95% intervals are clustered by video, and the reported intervals for the
three held-out models all cross zero. The release retains this limitation and
the post-hoc selection disclosure.

Rebuild the combined table from the frozen analysis files with:

```bash
python experiments/matched_pair/build_posthoc_natural50_model_panel.py \
  --selection-summary experiments/matched_pair/frozen_panel/candidate_summary.json \
  --candidate-pairs experiments/matched_pair/frozen_panel/candidate_pairs.csv \
  --expected-candidate-sha256 0756b51c9e734f02c5bbd801675f70a9816e9094ca96e16a351f4dfb9cb00e8f \
  --heldout Qwen3-VL-8B-Thinking=experiments/matched_pair/analyses/posthoc_natural50_qwen3vl8b_thinking_forced_analysis_20260812.json \
  --heldout Qwen3-VL-4B-Thinking=experiments/matched_pair/analyses/posthoc_natural50_qwen3vl4b_thinking_forced_analysis_20260812.json \
  --heldout Qwen2.5-VL-32B-Instruct-AWQ=experiments/matched_pair/analyses/posthoc_natural50_qwen25vl32b_awq_analysis_20260812.json \
  --sensitivity Qwen3-VL-8B-Thinking=experiments/matched_pair/analyses/posthoc_natural50_qwen3vl8b_thinking_raw_analysis_20260812.json \
  --sensitivity Qwen3-VL-4B-Thinking=experiments/matched_pair/analyses/posthoc_natural50_qwen3vl4b_thinking_raw_analysis_20260812.json \
  --output-json /tmp/matched_pair.json \
  --output-md /tmp/matched_pair.md
```
