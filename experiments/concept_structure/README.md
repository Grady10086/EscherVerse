# Concept Structure and Pair-Group Reanalysis

Protocol frozen: 2026-07-30

## Question

This experiment uses the submitted 27-model by six-category result table to
test whether category-level model performance is dominated by a broad common
factor, whether Action & Intent covaries with prerequisite spatial categories,
and what performance profile is obtained for the three category pairs shown in
the submitted overview figure.

It is an exploratory analysis of aggregate model results. It is not an
item-level test of teleological-spatial inseparability or of a
lower/intermediate/higher reasoning hierarchy.

## Locked Data

- Source: Table 1 in `sn-article.tex`.
- Source manuscript SHA-256:
  `c2af1b17cf3717a5240763fc05bb9331fef0be6903f737ed151970a3d93aed05`.
- Rows: 27 models.
- Measures: accuracy percentages for Categories 1--6 and overall accuracy.
- Locked copy: `data/model_category_scores.csv`.

No model is rerun for this experiment.

## Cohorts

1. `all_27`: primary analysis requested by Reviewer #1.
2. `non_sft_21`: removes six benchmark-trained SFT models.
3. `overall_ge_20_24`: sensitivity analysis removing three systems below 20%
   overall whose category profiles may be dominated by interface or answer
   formatting failure.
4. `non_sft_overall_ge_20_18`: combines the previous restrictions.

The threshold sensitivity analyses are reported, not substituted for the
primary 27-model result.

## Frozen Endpoints

- Pairwise Spearman correlations across the six category accuracies.
- Model-resampling 95% intervals for correlations.
- PCA of standardized category scores.
- Parallel analysis using independent within-column permutations.
- Leave-one-model-out stability of the first-component loadings.
- Equal-category-weight scores for the original figure-paired groups:
  - G1 Object continuity/state: Categories 1 and 5.
  - G2 Dynamic relation/prediction: Categories 2 and 4.
  - G3 Agent-intent/reference-frame: Categories 3 and 6.
- Item-count-weighted pair-group scores as a sensitivity analysis, using submitted
  category counts 1,086, 2,487, 662, 1,214, 211, and 2,335 after canonical
  label normalization. These counts cover 7,995 items; the five benchmark rows
  without a recoverable category label do not enter category weights.

G1/G2/G3 reproduce the two-by-two grouping in the submitted Figure 1. They are
descriptive groups on the figure's Object-Centric to Human-Centric continuum,
not lower/intermediate/higher hierarchy levels.

## Protocol Amendment

An earlier internal draft incorrectly assigned whole categories to three
hierarchy levels (`C1+C2+C5`, `C4+C6`, and `C3`). Re-auditing the submitted
article and Figure 1 showed that the submitted taxonomy instead pairs
`C1+C5`, `C2+C4`, and `C3+C6`. The hard category-to-level mapping was therefore
withdrawn before the corrected concept-structure result was drafted.

The proposed hierarchy describes item-level reasoning demand. A category such
as Action & Intent can include direct action recognition as well as
intent-conditioned prediction, and intent-conditioned items can occur outside
Category 3. Consequently, hierarchy labels require item-level annotation and
same-evidence controls; those analyses are addressed by the matched-pair study.

## Interpretation Rules

- A common first component supports shared performance variation across the
  current model panel, not a unitary human cognitive construct.
- Raw category correlations can reflect general model strength and do not by
  themselves establish that teleological and spatial reasoning are
  inseparable.
- Stable additional components support describing the categories as related
  but analytically distinguishable.
- Aggregate category scores cannot establish that individual questions jointly
  require teleological and spatial inference. Same-video matched pairs are
  handled in the matched-pair study.
- Differences among G1/G2/G3 describe the original figure-paired groups and
  must not be interpreted as empirical evidence for a difficulty hierarchy.
- Resampling intervals describe stability to the composition of this selected
  model panel; they are not population confidence intervals over all possible
  VLMs.

## Post-run Diagnostics

After the primary computation revealed a very strong first component, two
diagnostics were added before drafting the rebuttal:

1. rank-based PCA, whose correlation matrix is the Spearman matrix, to reduce
   sensitivity to the three very low-scoring systems;
2. leave-one-obvious-model-family-out PCA to check whether repeated sizes or
   Instruct/Thinking variants from one named family dominate the result.

These diagnostics were not part of the frozen primary endpoints. They are
reported transparently and do not replace the primary 27-model analysis.

## Run

```bash
python experiments/concept_structure/scripts/analyze_concept_structure.py \
  --scores experiments/concept_structure/data/model_category_scores.csv \
  --output-dir experiments/concept_structure/results \
  --seed 20260730 \
  --bootstrap 5000 \
  --parallel-permutations 5000
```

Run the unit tests:

```bash
python -m unittest tests.test_concept_structure
```
