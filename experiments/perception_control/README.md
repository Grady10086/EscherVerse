# Necessary-perception controls

The frozen manifest contains 200 benchmark items paired with a question about
a necessary visual prerequisite in the same sampled frames. The public result
files contain the seven-model paired summary and the stated max-token
sensitivity analysis. The `scripts/` directory includes sample construction,
probe validation, inference-manifest export, and paired analysis code.

Fresh model inference requires access to the corresponding clips. The public
summary can be checked without calling the models again:

- `final_pair_analysis.json`: primary scoring rule;
- `final_pair_analysis_parsed_sensitivity.json`: sensitivity analysis that
  accepts a parsed answer when generation reached its token limit.

Individual model outputs are not included in this compact public package, so
the public reproduction path is frozen-output inspection rather than a full seven-model
item-level rescore.
