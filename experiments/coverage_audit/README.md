# Coverage-bias audit

This directory contains the frozen sample, blind independent-authoring records,
historical-review attrition audit, and video-clustered coverage analysis.

The audit distinguishes directly observed counts from unrecoverable pipeline
stages. Historical handoff rows are not treated as independent candidate items
unless their normalized video-question key is unique.

## Recompute the final scope audit

The public mapping contains only blinded item IDs, author-run labels, and video
cluster IDs needed for the bootstrap. It contains no credentials or human
identity fields.

```bash
python experiments/coverage_audit/scripts/analyze_scope_audit.py \
  --packet experiments/coverage_audit/manifests/six_dimension_scope_packet_666.csv \
  --mapping experiments/coverage_audit/manifests/scope_mapping_666.csv \
  --rater-1 experiments/coverage_audit/judgments/scope_rater_1_all666.csv \
  --rater-2 experiments/coverage_audit/judgments/scope_rater_2_all666.csv \
  --output /tmp/scope_audit.json \
  --bootstrap 10000 \
  --seed 20260814
```
