# Experiment 1 Results

This report is generated from the locked 27-model Table 1 copy.

## Cohort sensitivity

| Cohort | n | Median off-diagonal Spearman rho | PC1 variance | Components above parallel 95% | G1 C1+C5 | G2 C2+C4 | G3 C3+C6 |
|---|---:|---:|---:|---:|---:|---:|---:|
| all_27 | 27 | 0.854 | 96.6% | 1 | 45.77 | 42.05 | 46.62 |
| non_sft_21 | 21 | 0.906 | 97.1% | 1 | 44.29 | 40.49 | 44.62 |
| overall_ge_20_24 | 24 | 0.793 | 86.6% | 1 | 50.39 | 45.17 | 50.27 |
| non_sft_overall_ge_20_18 | 18 | 0.852 | 89.2% | 1 | 50.20 | 44.38 | 49.14 |

## Primary Spearman correlation matrix

| | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---:|---:|---:|---:|---:|---:|
| C1 | 1.000 | 0.868 | 0.854 | 0.864 | 0.794 | 0.792 |
| C2 | 0.868 | 1.000 | 0.950 | 0.961 | 0.847 | 0.928 |
| C3 | 0.854 | 0.950 | 1.000 | 0.925 | 0.816 | 0.864 |
| C4 | 0.864 | 0.961 | 0.925 | 1.000 | 0.818 | 0.854 |
| C5 | 0.794 | 0.847 | 0.816 | 0.818 | 1.000 | 0.753 |
| C6 | 0.792 | 0.928 | 0.864 | 0.854 | 0.753 | 1.000 |

Category order: C1 Object permanence; C2 Dynamic spatial; C3 Action & intent; C4 Predictive/counterfactual; C5 Deformation/state; C6 Ego/allo.

## Action & Intent correlations

| Comparison | rho | Model-resampling interval |
|---|---:|---:|
| C3 vs C1 | 0.854 | [0.654, 0.945] |
| C3 vs C2 | 0.950 | [0.840, 0.986] |
| C3 vs C4 | 0.925 | [0.797, 0.976] |
| C3 vs C5 | 0.816 | [0.563, 0.938] |
| C3 vs C6 | 0.864 | [0.651, 0.963] |

## PCA and parallel analysis

| Component | Observed eigenvalue | Explained variance | Parallel 95% threshold |
|---|---:|---:|---:|
| PC1 | 5.793 | 96.6% | 2.024 |
| PC2 | 0.082 | 1.4% | 1.516 |
| PC3 | 0.072 | 1.2% | 1.217 |
| PC4 | 0.035 | 0.6% | 0.984 |
| PC5 | 0.014 | 0.2% | 0.786 |
| PC6 | 0.003 | 0.1% | 0.590 |

### Primary PCA loadings

| Category | PC1 | PC2 |
|---|---:|---:|
| C1 Object permanence | 0.979 | 0.116 |
| C2 Dynamic spatial | 0.996 | 0.014 |
| C3 Action & intent | 0.977 | -0.200 |
| C4 Predictive/counterfactual | 0.991 | -0.090 |
| C5 Deformation/state | 0.970 | 0.019 |
| C6 Ego/allo | 0.983 | 0.142 |

## Original figure-paired group profile

| Pair group | Mean | Median | Model-resampling interval for mean | Min | Max |
|---|---:|---:|---:|---:|---:|
| G1 Object continuity/state (C1+C5) | 45.77 | 50.58 | [39.75, 50.78] | 4.55 | 64.64 |
| G2 Dynamic relation/prediction (C2+C4) | 42.05 | 43.95 | [38.03, 45.40] | 13.16 | 55.75 |
| G3 Agent-intent/reference-frame (C3+C6) | 46.62 | 49.27 | [41.89, 50.54] | 11.29 | 61.96 |

## Post-run robustness diagnostics

- Rank-based PC1 variance: 88.4%.
- Leave-one-named-family-out PC1 variance range: 95.6% to 97.8%.
- These diagnostics were added after the primary result and are not substituted for it.

## Interpretation boundary

- The model panel is purposively selected, not a random sample of all VLMs.
- Raw category correlations can be driven by broad model strength.
- Category aggregates cannot establish item-level inseparability.
- G1/G2/G3 reproduce the three color-paired groups in the submitted overview figure; they are not hierarchy levels.
- Reviewer #2's hierarchy requires item-level labels and is deferred to Experiment 2.
