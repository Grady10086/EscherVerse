# Quantitative failure-mode analysis

The release includes the 300-item stratified pool, two completed blinded review
files, the completed adjudication file, and the final weighted analysis. Model
identity is not included in the rater interface fields used for labeling.

Recompute the reliability and weighted frequencies with:

```bash
python experiments/failure_mode/scripts/analyze_annotations.py \
  --rater-1 experiments/failure_mode/annotation_pool_300/rater_1.csv \
  --rater-2 experiments/failure_mode/annotation_pool_300/rater_2.csv \
  --audit-manifest experiments/failure_mode/annotation_pool_300/audit_manifest.json \
  --adjudicated experiments/failure_mode/final_annotations/adjudication_final.csv \
  --expected-rows 300 \
  --output-dir /tmp/failure_mode
```
