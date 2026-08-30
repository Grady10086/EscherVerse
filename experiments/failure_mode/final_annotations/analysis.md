# Failure-mode Annotation Analysis

## Reliability

- Analysis units: 300 model-item errors.
- Primary-label raw agreement: 79.3%.
- Cohen's kappa: 0.7451077125472784 (expected agreement 18.9%).
- Question/key-problem raw agreement: 89.3%.
- Secondary-label exact agreement: 63.7%.

Kappa is reported alongside rater-specific label prevalence in the JSON output because rare labels can depress kappa despite high raw agreement.

## Adjudicated Primary-label Frequencies

| Label | Unweighted n (%) | Weighted % | Unweighted Wilson 95% CI | Weighted Wilson 95% CI* |
|---|---:|---:|---:|---:|
| benchmark_ambiguity | 29 (9.7%) | 9.6% | 6.8% to 13.5% | 6.8% to 13.5% |
| protocol_or_parsing | 12 (4.0%) | 4.1% | 2.3% to 6.9% | 2.4% to 7.0% |
| perception_recognition | 36 (12.0%) | 12.1% | 8.8% to 16.2% | 8.9% to 16.3% |
| spatiotemporal_grounding | 86 (28.7%) | 28.8% | 23.8% to 34.0% | 24.0% to 34.2% |
| perspective_reference_frame | 72 (24.0%) | 23.7% | 19.5% to 29.1% | 19.3% to 28.9% |
| action_goal_binding | 22 (7.3%) | 7.4% | 4.9% to 10.9% | 5.0% to 11.0% |
| physical_prediction_counterfactual | 43 (14.3%) | 14.2% | 10.8% to 18.8% | 10.7% to 18.6% |
| other_unresolved | 0 (0.0%) | 0.0% | 0.0% to 1.3% | 0.0% to 1.3% |

*Weighted point estimates use inverse inclusion-probability weights. The weighted Wilson interval uses Kish effective n=299.3; it is descriptive rather than a design-based confidence interval.

## Scope and Inference

The sampling and reporting unit is a model-item error. The 300 units map to 169 benchmark items, and 98 items occur for more than one model. Those repeated items create within-item dependence. Therefore Wilson intervals are descriptive summaries of this stratified error sample, not independent-unit inferential tests or model-comparison p-values.

The JSON output additionally contains model, six-category, and control-pass strata; a multi-label secondary-label sensitivity analysis; and exclusions for benchmark ambiguity alone and for benchmark ambiguity plus protocol/parsing cases.
