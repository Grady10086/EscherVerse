from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


CATEGORY_COLUMNS = [
    "c1_object_permanence",
    "c2_dynamic_spatial",
    "c3_action_intent",
    "c4_predictive_counterfactual",
    "c5_deformation_state",
    "c6_ego_allo",
]

CATEGORY_LABELS = [
    "C1 Object permanence",
    "C2 Dynamic spatial",
    "C3 Action & intent",
    "C4 Predictive/counterfactual",
    "C5 Deformation/state",
    "C6 Ego/allo",
]

PAIR_GROUP_LABELS = [
    "G1 Object continuity/state (C1+C5)",
    "G2 Dynamic relation/prediction (C2+C4)",
    "G3 Agent-intent/reference-frame (C3+C6)",
]

PAIR_GROUPS = ([0, 4], [1, 3], [2, 5])

CATEGORY_COUNTS = np.asarray([1086, 2487, 662, 1214, 211, 2335], dtype=float)


def load_scores(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"model", "model_group", "training_regime", "overall", *CATEGORY_COLUMNS}
    if not rows:
        raise ValueError("Score table is empty")
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"Score table is missing columns: {sorted(missing)}")
    if len(rows) != 27:
        raise ValueError(f"Expected the submitted 27-model panel, found {len(rows)}")
    if len({row["model"] for row in rows}) != len(rows):
        raise ValueError("Model names must be unique")
    for row in rows:
        for column in [*CATEGORY_COLUMNS, "overall"]:
            value = float(row[column])
            if not 0.0 <= value <= 100.0:
                raise ValueError(f"{row['model']} has invalid {column}: {value}")
            row[column] = value
    return rows


def score_matrix(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray(
        [[float(row[column]) for column in CATEGORY_COLUMNS] for row in rows],
        dtype=float,
    )


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def spearman_matrix(values: np.ndarray) -> np.ndarray:
    ranked = np.column_stack([rankdata(values[:, index]) for index in range(values.shape[1])])
    return np.corrcoef(ranked, rowvar=False)


def rank_transform(values: np.ndarray) -> np.ndarray:
    return np.column_stack(
        [rankdata(values[:, index]) for index in range(values.shape[1])]
    )


def pca(values: np.ndarray) -> dict[str, np.ndarray]:
    standard_deviation = values.std(axis=0, ddof=1)
    if np.any(standard_deviation == 0):
        raise ValueError("PCA requires non-constant category scores")
    standardized = (values - values.mean(axis=0)) / standard_deviation
    correlation = np.corrcoef(standardized, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(correlation)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    loadings = eigenvectors * np.sqrt(np.clip(eigenvalues, 0.0, None))
    for component in range(loadings.shape[1]):
        if loadings[:, component].sum() < 0:
            eigenvectors[:, component] *= -1
            loadings[:, component] *= -1
    return {
        "correlation": correlation,
        "eigenvalues": eigenvalues,
        "explained_ratio": eigenvalues / eigenvalues.sum(),
        "eigenvectors": eigenvectors,
        "loadings": loadings,
    }


def bootstrap_correlations(
    values: np.ndarray,
    *,
    rng: np.random.Generator,
    iterations: int,
) -> dict[str, list[float]]:
    samples: dict[str, list[float]] = {
        f"c{left + 1}_c{right + 1}": []
        for left in range(values.shape[1])
        for right in range(left + 1, values.shape[1])
    }
    for _ in range(iterations):
        indices = rng.integers(0, len(values), size=len(values))
        sampled = values[indices]
        if np.any(sampled.std(axis=0) == 0):
            continue
        correlation = spearman_matrix(sampled)
        for left in range(values.shape[1]):
            for right in range(left + 1, values.shape[1]):
                value = correlation[left, right]
                if np.isfinite(value):
                    samples[f"c{left + 1}_c{right + 1}"].append(float(value))
    return samples


def parallel_analysis(
    values: np.ndarray,
    *,
    rng: np.random.Generator,
    iterations: int,
) -> np.ndarray:
    null_eigenvalues = np.empty((iterations, values.shape[1]), dtype=float)
    for iteration in range(iterations):
        permuted = np.column_stack(
            [rng.permutation(values[:, column]) for column in range(values.shape[1])]
        )
        null_eigenvalues[iteration] = pca(permuted)["eigenvalues"]
    return np.quantile(null_eigenvalues, 0.95, axis=0)


def leave_one_out_pc1(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    loadings = []
    for omitted in range(len(values)):
        subset = np.delete(values, omitted, axis=0)
        current = pca(subset)["loadings"][:, 0]
        if np.dot(current, reference) < 0:
            current *= -1
        loadings.append(current)
    return np.asarray(loadings)


def named_model_family(model: str) -> str:
    rules = (
        ("Qwen", ("Qwen", "qwen")),
        ("LLaVA", ("LLaVA",)),
        ("MiniCPM", ("MiniCPM",)),
        ("InternVL", ("InternVL",)),
        ("ERNIE", ("ERNIE",)),
        ("Spatial-MLLM", ("Spatial-MLLM",)),
        ("ViCA", ("ViCA",)),
        ("ISI", ("ISI",)),
        ("Escher", ("Escher",)),
        ("GPT", ("GPT",)),
        ("Gemini", ("Gemini",)),
    )
    for family, prefixes in rules:
        if model.startswith(prefixes):
            return family
    raise ValueError(f"No named-family rule for model: {model}")


def leave_one_family_out_pca(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    families = sorted({named_model_family(row["model"]) for row in rows})
    records = []
    for family in families:
        remaining = [
            row for row in rows if named_model_family(row["model"]) != family
        ]
        result = pca(score_matrix(remaining))
        records.append(
            {
                "omitted_family": family,
                "n_models": len(remaining),
                "pc1_explained_ratio": float(result["explained_ratio"][0]),
                "pc1_loadings": result["loadings"][:, 0].tolist(),
            }
        )
    return records


def pair_group_scores(values: np.ndarray, weighted: bool = False) -> np.ndarray:
    columns = []
    for group in PAIR_GROUPS:
        if weighted:
            columns.append(
                np.average(values[:, group], axis=1, weights=CATEGORY_COUNTS[group])
            )
        else:
            columns.append(values[:, group].mean(axis=1))
    return np.column_stack(columns)


def model_pair_group_records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    equal_weight = pair_group_scores(score_matrix(rows))
    item_weighted = pair_group_scores(score_matrix(rows), weighted=True)
    records = []
    for index, row in enumerate(rows):
        records.append(
            {
                "model": row["model"],
                "model_group": row["model_group"],
                "training_regime": row["training_regime"],
                "group_1_c1_c5_equal_category_weight": float(equal_weight[index, 0]),
                "group_2_c2_c4_equal_category_weight": float(equal_weight[index, 1]),
                "group_3_c3_c6_equal_category_weight": float(equal_weight[index, 2]),
                "group_1_c1_c5_item_count_weighted": float(item_weighted[index, 0]),
                "group_2_c2_c4_item_count_weighted": float(item_weighted[index, 1]),
                "group_3_c3_c6_item_count_weighted": float(item_weighted[index, 2]),
                "overall": float(row["overall"]),
            }
        )
    return records


def interval(values: list[float] | np.ndarray) -> list[float]:
    array = np.asarray(values, dtype=float)
    return [float(np.quantile(array, 0.025)), float(np.quantile(array, 0.975))]


def bootstrap_pair_group_summary(
    group_scores: np.ndarray,
    *,
    rng: np.random.Generator,
    iterations: int,
) -> dict[str, Any]:
    sampled_means = np.empty((iterations, group_scores.shape[1]), dtype=float)
    differences = {
        "g2_minus_g1": np.empty(iterations, dtype=float),
        "g3_minus_g2": np.empty(iterations, dtype=float),
        "g3_minus_g1": np.empty(iterations, dtype=float),
    }
    for iteration in range(iterations):
        indices = rng.integers(0, len(group_scores), size=len(group_scores))
        current = group_scores[indices]
        sampled_means[iteration] = current.mean(axis=0)
        differences["g2_minus_g1"][iteration] = np.mean(current[:, 1] - current[:, 0])
        differences["g3_minus_g2"][iteration] = np.mean(current[:, 2] - current[:, 1])
        differences["g3_minus_g1"][iteration] = np.mean(current[:, 2] - current[:, 0])
    summaries = []
    for index, label in enumerate(PAIR_GROUP_LABELS):
        summaries.append(
            {
                "group": label,
                "mean": float(group_scores[:, index].mean()),
                "median": float(np.median(group_scores[:, index])),
                "model_resampling_interval_for_mean": interval(sampled_means[:, index]),
                "minimum": float(group_scores[:, index].min()),
                "maximum": float(group_scores[:, index].max()),
            }
        )
    return {
        "groups": summaries,
        "mean_differences": {
            key: {
                "estimate": float(
                    {
                        "g2_minus_g1": np.mean(group_scores[:, 1] - group_scores[:, 0]),
                        "g3_minus_g2": np.mean(group_scores[:, 2] - group_scores[:, 1]),
                        "g3_minus_g1": np.mean(group_scores[:, 2] - group_scores[:, 0]),
                    }[key]
                ),
                "model_resampling_interval": interval(values),
                "models_with_positive_difference": int(
                    {
                        "g2_minus_g1": np.sum(group_scores[:, 1] > group_scores[:, 0]),
                        "g3_minus_g2": np.sum(group_scores[:, 2] > group_scores[:, 1]),
                        "g3_minus_g1": np.sum(group_scores[:, 2] > group_scores[:, 0]),
                    }[key]
                ),
                "models_total": int(len(group_scores)),
            }
            for key, values in differences.items()
        },
    }


def cohort_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    return {
        "all_27": rows,
        "non_sft_21": [row for row in rows if row["training_regime"] != "sft"],
        "overall_ge_20_24": [row for row in rows if float(row["overall"]) >= 20.0],
        "non_sft_overall_ge_20_18": [
            row
            for row in rows
            if row["training_regime"] != "sft" and float(row["overall"]) >= 20.0
        ],
    }


def analyze_cohort(
    rows: list[dict[str, Any]],
    *,
    seed: int,
    bootstrap_iterations: int,
    parallel_iterations: int,
) -> dict[str, Any]:
    values = score_matrix(rows)
    correlation = spearman_matrix(values)
    rng = np.random.default_rng(seed)
    bootstrapped = bootstrap_correlations(
        values,
        rng=rng,
        iterations=bootstrap_iterations,
    )
    pca_result = pca(values)
    rank_pca_result = pca(rank_transform(values))
    null_95 = parallel_analysis(
        values,
        rng=rng,
        iterations=parallel_iterations,
    )
    loo = leave_one_out_pc1(values, pca_result["loadings"][:, 0])
    off_diagonal = correlation[np.triu_indices(values.shape[1], k=1)]
    correlation_records = {}
    for left in range(values.shape[1]):
        for right in range(left + 1, values.shape[1]):
            key = f"c{left + 1}_c{right + 1}"
            correlation_records[key] = {
                "rho": float(correlation[left, right]),
                "model_resampling_interval": interval(bootstrapped[key]),
            }
    return {
        "n_models": len(rows),
        "models": [row["model"] for row in rows],
        "spearman_correlation": correlation.tolist(),
        "correlation_records": correlation_records,
        "median_off_diagonal_rho": float(np.median(off_diagonal)),
        "pca": {
            "eigenvalues": pca_result["eigenvalues"].tolist(),
            "explained_ratio": pca_result["explained_ratio"].tolist(),
            "loadings": pca_result["loadings"].tolist(),
            "parallel_analysis_95th_percentile": null_95.tolist(),
            "components_above_parallel_95": int(
                np.sum(pca_result["eigenvalues"] > null_95)
            ),
            "pc1_leave_one_out_loading_min": loo.min(axis=0).tolist(),
            "pc1_leave_one_out_loading_max": loo.max(axis=0).tolist(),
        },
        "post_run_diagnostics": {
            "rank_based_pca": {
                "eigenvalues": rank_pca_result["eigenvalues"].tolist(),
                "explained_ratio": rank_pca_result["explained_ratio"].tolist(),
                "loadings": rank_pca_result["loadings"].tolist(),
            },
            "leave_one_named_family_out": leave_one_family_out_pca(rows),
        },
        "original_pair_groups_equal_category_weight": bootstrap_pair_group_summary(
            pair_group_scores(values),
            rng=rng,
            iterations=bootstrap_iterations,
        ),
        "original_pair_groups_item_count_weighted_sensitivity": bootstrap_pair_group_summary(
            pair_group_scores(values, weighted=True),
            rng=rng,
            iterations=bootstrap_iterations,
        ),
    }


def analyze(
    rows: list[dict[str, Any]],
    *,
    seed: int,
    bootstrap_iterations: int,
    parallel_iterations: int,
) -> dict[str, Any]:
    results = {}
    for index, (name, current_rows) in enumerate(cohort_rows(rows).items()):
        results[name] = analyze_cohort(
            current_rows,
            seed=seed + index * 1000,
            bootstrap_iterations=bootstrap_iterations,
            parallel_iterations=parallel_iterations,
        )
    return {
        "metadata": {
            "analysis": "concept_structure_and_original_pair_groups",
            "seed": seed,
            "bootstrap_iterations": bootstrap_iterations,
            "parallel_permutations": parallel_iterations,
            "source_manuscript_sha256": (
                "c2af1b17cf3717a5240763fc05bb9331fef0be6903f737ed151970a3d93aed05"
            ),
            "interpretation": (
                "Exploratory aggregate-model analysis; model-resampling intervals "
                "are not population confidence intervals."
            ),
        },
        "cohorts": results,
    }


def markdown_matrix(matrix: list[list[float]]) -> str:
    header = "| | " + " | ".join(f"C{index}" for index in range(1, 7)) + " |"
    divider = "|---|" + "|".join(["---:"] * 6) + "|"
    rows = [header, divider]
    for index, values in enumerate(matrix, start=1):
        rows.append(
            f"| C{index} | " + " | ".join(f"{value:.3f}" for value in values) + " |"
        )
    return "\n".join(rows)


def render_markdown(results: dict[str, Any]) -> str:
    primary = results["cohorts"]["all_27"]
    lines = [
        "# Experiment 1 Results",
        "",
        "This report is generated from the locked 27-model Table 1 copy.",
        "",
        "## Cohort sensitivity",
        "",
        "| Cohort | n | Median off-diagonal Spearman rho | PC1 variance | Components above parallel 95% | G1 C1+C5 | G2 C2+C4 | G3 C3+C6 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, cohort in results["cohorts"].items():
        groups = cohort["original_pair_groups_equal_category_weight"]["groups"]
        lines.append(
            f"| {name} | {cohort['n_models']} | "
            f"{cohort['median_off_diagonal_rho']:.3f} | "
            f"{100 * cohort['pca']['explained_ratio'][0]:.1f}% | "
            f"{cohort['pca']['components_above_parallel_95']} | "
            f"{groups[0]['mean']:.2f} | {groups[1]['mean']:.2f} | "
            f"{groups[2]['mean']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Primary Spearman correlation matrix",
            "",
            markdown_matrix(primary["spearman_correlation"]),
            "",
            "Category order: " + "; ".join(CATEGORY_LABELS) + ".",
            "",
            "## Action & Intent correlations",
            "",
            "| Comparison | rho | Model-resampling interval |",
            "|---|---:|---:|",
        ]
    )
    for other in [0, 1, 3, 4, 5]:
        left, right = sorted((2, other))
        record = primary["correlation_records"][f"c{left + 1}_c{right + 1}"]
        low, high = record["model_resampling_interval"]
        lines.append(
            f"| C3 vs C{other + 1} | {record['rho']:.3f} | "
            f"[{low:.3f}, {high:.3f}] |"
        )
    lines.extend(
        [
            "",
            "## PCA and parallel analysis",
            "",
            "| Component | Observed eigenvalue | Explained variance | Parallel 95% threshold |",
            "|---|---:|---:|---:|",
        ]
    )
    for index, (eigenvalue, ratio, threshold) in enumerate(
        zip(
            primary["pca"]["eigenvalues"],
            primary["pca"]["explained_ratio"],
            primary["pca"]["parallel_analysis_95th_percentile"],
        ),
        start=1,
    ):
        lines.append(
            f"| PC{index} | {eigenvalue:.3f} | {100 * ratio:.1f}% | "
            f"{threshold:.3f} |"
        )
    lines.extend(
        [
            "",
            "### Primary PCA loadings",
            "",
            "| Category | PC1 | PC2 |",
            "|---|---:|---:|",
        ]
    )
    for label, loadings in zip(CATEGORY_LABELS, primary["pca"]["loadings"]):
        lines.append(f"| {label} | {loadings[0]:.3f} | {loadings[1]:.3f} |")
    lines.extend(
        [
            "",
            "## Original figure-paired group profile",
            "",
            "| Pair group | Mean | Median | Model-resampling interval for mean | Min | Max |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for record in primary["original_pair_groups_equal_category_weight"]["groups"]:
        low, high = record["model_resampling_interval_for_mean"]
        lines.append(
            f"| {record['group']} | {record['mean']:.2f} | "
            f"{record['median']:.2f} | [{low:.2f}, {high:.2f}] | "
            f"{record['minimum']:.2f} | {record['maximum']:.2f} |"
        )
    family_diagnostics = primary["post_run_diagnostics"][
        "leave_one_named_family_out"
    ]
    family_pc1 = [record["pc1_explained_ratio"] for record in family_diagnostics]
    lines.extend(
        [
            "",
            "## Post-run robustness diagnostics",
            "",
            f"- Rank-based PC1 variance: "
            f"{100 * primary['post_run_diagnostics']['rank_based_pca']['explained_ratio'][0]:.1f}%.",
            f"- Leave-one-named-family-out PC1 variance range: "
            f"{100 * min(family_pc1):.1f}% to {100 * max(family_pc1):.1f}%.",
            "- These diagnostics were added after the primary result and are not substituted for it.",
        ]
    )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "- The model panel is purposively selected, not a random sample of all VLMs.",
            "- Raw category correlations can be driven by broad model strength.",
            "- Category aggregates cannot establish item-level inseparability.",
            "- G1/G2/G3 reproduce the three color-paired groups in the submitted overview figure; they are not hierarchy levels.",
            "- Reviewer #2's hierarchy requires item-level labels and is deferred to Experiment 2.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--parallel-permutations", type=int, default=5000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_scores(args.scores)
    results = analyze(
        rows,
        seed=args.seed,
        bootstrap_iterations=args.bootstrap,
        parallel_iterations=args.parallel_permutations,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "analysis_results.json").write_text(
        json.dumps(results, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "analysis_results.md").write_text(
        render_markdown(results),
        encoding="utf-8",
    )
    pair_group_records = model_pair_group_records(rows)
    csv_records = [
        {
            key: f"{value:.2f}" if isinstance(value, float) else value
            for key, value in record.items()
        }
        for record in pair_group_records
    ]
    with (args.output_dir / "model_pair_group_scores.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_records[0]))
        writer.writeheader()
        writer.writerows(csv_records)
    print(render_markdown(results))


if __name__ == "__main__":
    main()
