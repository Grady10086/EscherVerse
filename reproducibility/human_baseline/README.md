# Human first-pass baseline

The manuscript reports 11 annotators, 8,000 first-pass judgments per annotator,
a mean accuracy of 90.62%, and a range of 84.81%-95.14%.

`aggregate.py` defines the public aggregation procedure. Its input is an
anonymized CSV with one row per judgment and two required columns:
`annotator_id` and `is_correct`. It refuses incomplete annotator panels by
default.

`reported_summary.json` is an aggregate-only validation record. Item-level
first-pass judgments are not included in the public release. This path verifies
the manuscript-level aggregate and aggregation contract but does not
independently reconstruct the result from item-level judgments.
