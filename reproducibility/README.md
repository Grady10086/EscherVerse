# Reproducibility guide

This directory documents the parts of the paper that can be rerun publicly and
the parts that can only be checked from frozen outputs or aggregates.

## Levels

- **Fresh inference:** rerun with available videos and a supported model.
- **Frozen-output recomputation:** rerun scoring or statistics without calling
  the original model again.
- **Aggregate verification:** inspect a released summary when media, model, or
  privacy restrictions prevent release of the underlying record.

The model-by-model status is in `models/model_registry.csv`. Video
reconstruction is in `video/`, the released LoRA entry point is in `training/`,
and the human first-pass aggregation contract is in `human_baseline/`.

The public release does not claim that mutable closed APIs reproduce historical
outputs exactly, that removed third-party videos remain recoverable, or that an
aggregate-only row is an end-to-end reproducible result.
