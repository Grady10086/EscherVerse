#!/usr/bin/env python3
"""Analyze blinded dual-rater annotations and adjudicated failure labels.

This module intentionally never discovers annotation files on its own.  The caller
must provide explicit CSV paths, so it is safe to use before the blinded rater
files are frozen.  It performs two stages:

1. validate the two completed rater files, estimate reliability, and write a
   blinded disagreement queue for adjudication;
2. after a completed queue is supplied, merge adjudications and write the
   design-weighted and unweighted descriptive frequency report.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import NormalDist
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST = (
    ROOT
    / "experiments/failure_mode/annotation_pool_300/audit_manifest.json"
)

LABELS = (
    "benchmark_ambiguity",
    "protocol_or_parsing",
    "perception_recognition",
    "spatiotemporal_grounding",
    "perspective_reference_frame",
    "action_goal_binding",
    "physical_prediction_counterfactual",
    "other_unresolved",
)
CONFIDENCES = {"high", "medium", "low"}
BOOLEAN_VALUES = {"yes": True, "true": True, "1": True, "no": False, "false": False, "0": False}
ANNOTATION_FIELDS = {
    "primary_label",
    "secondary_label",
    "evidence_note",
    "confidence",
    "question_or_key_problem",
}
RATER_REQUIRED_FIELDS = {
    "unit_id",
    "primary_label",
    "secondary_label",
    "evidence_note",
    "confidence",
    "question_or_key_problem",
}
IMMUTABLE_RATER_FIELDS = (
    "video",
    "contact_sheet",
    "category_id",
    "category",
    "scene_type",
    "question_type",
    "question",
    "gold_answer",
    "model_answer",
    "control_type",
    "control_question",
    "control_gold_answer",
    "control_model_answer",
    "control_pass",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def normalized_text(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return "" if value is None else str(value).strip()


def parse_boolean(value: Any, field: str, unit_id: str) -> bool:
    normalized = normalized_text(value).lower()
    if normalized not in BOOLEAN_VALUES:
        raise ValueError(f"{field} must be yes/no (or true/false) for {unit_id}; got {value!r}")
    return BOOLEAN_VALUES[normalized]


def load_audit(path: Path, expected_rows: int) -> dict[str, dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or len(rows) != expected_rows:
        raise ValueError(f"Audit manifest must contain exactly {expected_rows} rows")
    by_id = {str(row.get("unit_id", "")): row for row in rows}
    if "" in by_id or len(by_id) != expected_rows:
        raise ValueError("Audit manifest has missing or duplicate unit_id values")
    required = {"model", "benchmark_index", "category_id", "category", "control_pass", "analysis_weight"}
    for unit_id, row in by_id.items():
        missing = required - set(row)
        if missing:
            raise ValueError(f"Audit row {unit_id} is missing: {sorted(missing)}")
        if float(row["analysis_weight"]) <= 0:
            raise ValueError(f"Audit row {unit_id} has non-positive analysis_weight")
    return by_id


def validate_rater_rows(
    rows: list[dict[str, str]],
    audit: dict[str, dict[str, Any]],
    rater_name: str,
    expected_rows: int,
) -> dict[str, dict[str, Any]]:
    if len(rows) != expected_rows:
        raise ValueError(f"{rater_name} must contain exactly {expected_rows} rows; found {len(rows)}")
    if not rows:
        raise ValueError(f"{rater_name} is empty")
    missing_headers = RATER_REQUIRED_FIELDS - set(rows[0])
    if missing_headers:
        raise ValueError(f"{rater_name} is missing required columns: {sorted(missing_headers)}")
    if "model" in rows[0]:
        raise ValueError(f"{rater_name} is not blinded: it contains a model column")

    mapped: dict[str, dict[str, Any]] = {}
    for row_number, raw in enumerate(rows, start=2):
        unit_id = normalized_text(raw.get("unit_id"))
        if not unit_id or unit_id in mapped:
            raise ValueError(f"{rater_name} has missing or duplicate unit_id at CSV row {row_number}")
        if unit_id not in audit:
            raise ValueError(f"{rater_name} has unit_id absent from audit manifest: {unit_id}")
        missing_row_fields = set(IMMUTABLE_RATER_FIELDS) - set(raw)
        if missing_row_fields:
            raise ValueError(f"{rater_name} is missing immutable columns: {sorted(missing_row_fields)}")
        for field in IMMUTABLE_RATER_FIELDS:
            rater_value = normalized_text(raw[field])
            audit_value = normalized_text(audit[unit_id].get(field))
            if field == "control_pass":
                rater_value = rater_value.lower()
                audit_value = audit_value.lower()
            if rater_value != audit_value:
                raise ValueError(f"{rater_name} immutable field {field!r} does not match audit for {unit_id}")

        primary = normalized_text(raw.get("primary_label"))
        secondary = normalized_text(raw.get("secondary_label"))
        if primary not in LABELS:
            raise ValueError(f"{rater_name} has invalid primary_label for {unit_id}: {primary!r}")
        if secondary and secondary not in LABELS:
            raise ValueError(f"{rater_name} has invalid secondary_label for {unit_id}: {secondary!r}")
        if secondary == primary:
            raise ValueError(f"{rater_name} secondary_label must differ from primary_label for {unit_id}")
        if normalized_text(raw.get("confidence")).lower() not in CONFIDENCES:
            raise ValueError(f"{rater_name} has invalid confidence for {unit_id}")
        if not normalized_text(raw.get("evidence_note")):
            raise ValueError(f"{rater_name} evidence_note is empty for {unit_id}")
        mapped[unit_id] = {
            "primary_label": primary,
            "secondary_label": secondary,
            "confidence": normalized_text(raw["confidence"]).lower(),
            "question_or_key_problem": parse_boolean(raw["question_or_key_problem"], "question_or_key_problem", unit_id),
            "evidence_note": normalized_text(raw["evidence_note"]),
            "source_row": raw,
        }
    if set(mapped) != set(audit):
        raise ValueError(f"{rater_name} unit IDs do not exactly align with the audit manifest")
    return mapped


def label_prevalence(rows: Iterable[dict[str, Any]], field: str = "primary_label") -> dict[str, dict[str, float | int]]:
    rows = list(rows)
    count = len(rows)
    frequencies = Counter(row[field] for row in rows)
    return {
        label: {"count": frequencies[label], "proportion": frequencies[label] / count if count else None}
        for label in LABELS
    }


def cohen_kappa(first: Iterable[str], second: Iterable[str]) -> dict[str, float | None]:
    first, second = list(first), list(second)
    if len(first) != len(second) or not first:
        raise ValueError("Cohen kappa requires equal, non-empty vectors")
    n = len(first)
    observed = sum(a == b for a, b in zip(first, second)) / n
    first_counts, second_counts = Counter(first), Counter(second)
    expected = sum((first_counts[label] / n) * (second_counts[label] / n) for label in LABELS)
    denominator = 1.0 - expected
    return {
        "observed_agreement": observed,
        "expected_agreement": expected,
        "kappa": None if math.isclose(denominator, 0.0) else (observed - expected) / denominator,
    }


def build_disagreement_queue(
    rater_1: dict[str, dict[str, Any]], rater_2: dict[str, dict[str, Any]], output: Path
) -> list[str]:
    unit_ids: list[str] = []
    rows: list[dict[str, Any]] = []
    context_fields = [field for field in IMMUTABLE_RATER_FIELDS if field != "contact_sheet"]
    for unit_id in sorted(rater_1):
        one, two = rater_1[unit_id], rater_2[unit_id]
        reasons = []
        if one["primary_label"] != two["primary_label"]:
            reasons.append("primary_label")
        if one["secondary_label"] != two["secondary_label"]:
            reasons.append("secondary_label")
        if one["question_or_key_problem"] != two["question_or_key_problem"]:
            reasons.append("question_or_key_problem")
        if not reasons:
            continue
        source = one["source_row"]
        row: dict[str, Any] = {"unit_id": unit_id, "disagreement_fields": ";".join(reasons)}
        row.update({field: source[field] for field in context_fields})
        for prefix, annotation in (("rater_1", one), ("rater_2", two)):
            for field in ("primary_label", "secondary_label", "confidence", "question_or_key_problem", "evidence_note"):
                row[f"{prefix}_{field}"] = annotation[field]
        row.update(
            adjudicated_primary_label="",
            adjudicated_secondary_label="",
            adjudicated_question_or_key_problem="",
            adjudication_note="",
        )
        unit_ids.append(unit_id)
        rows.append(row)
    fields = ["unit_id", "disagreement_fields", *context_fields]
    fields += [
        f"{prefix}_{field}"
        for prefix in ("rater_1", "rater_2")
        for field in ("primary_label", "secondary_label", "confidence", "question_or_key_problem", "evidence_note")
    ]
    fields += [
        "adjudicated_primary_label",
        "adjudicated_secondary_label",
        "adjudicated_question_or_key_problem",
        "adjudication_note",
    ]
    write_csv(output, rows, fields)
    return unit_ids


def load_adjudications(
    path: Path, eligible_ids: set[str], rater_1: dict[str, dict[str, Any]], rater_2: dict[str, dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    rows = read_csv(path)
    if not rows:
        if eligible_ids:
            raise ValueError("Adjudication CSV is empty but disagreements require adjudication")
        return {}
    required = {"unit_id", "adjudicated_primary_label", "adjudicated_secondary_label", "adjudicated_question_or_key_problem"}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"Adjudication CSV missing columns: {sorted(missing)}")
    result: dict[str, dict[str, Any]] = {}
    for number, row in enumerate(rows, start=2):
        unit_id = normalized_text(row.get("unit_id"))
        if not unit_id or unit_id in result or unit_id not in eligible_ids:
            raise ValueError(f"Invalid, duplicate, or non-disagreement adjudication unit_id at CSV row {number}: {unit_id!r}")
        primary = normalized_text(row.get("adjudicated_primary_label"))
        secondary = normalized_text(row.get("adjudicated_secondary_label"))
        question_problem = normalized_text(row.get("adjudicated_question_or_key_problem"))
        primary_disagrees = rater_1[unit_id]["primary_label"] != rater_2[unit_id]["primary_label"]
        question_disagrees = rater_1[unit_id]["question_or_key_problem"] != rater_2[unit_id]["question_or_key_problem"]
        if primary_disagrees and primary not in LABELS:
            raise ValueError(f"A primary-label disagreement needs a valid adjudicated label for {unit_id}")
        if primary and primary not in LABELS:
            raise ValueError(f"Invalid adjudicated primary label for {unit_id}: {primary!r}")
        if secondary and secondary not in LABELS:
            raise ValueError(f"Invalid adjudicated secondary label for {unit_id}: {secondary!r}")
        final_primary = primary or rater_1[unit_id]["primary_label"]
        if secondary == final_primary:
            raise ValueError(f"Adjudicated secondary label must differ from final primary label for {unit_id}")
        if question_disagrees and normalized_text(question_problem).lower() not in BOOLEAN_VALUES:
            raise ValueError(f"A question/key disagreement needs an adjudicated yes/no value for {unit_id}")
        if question_problem and normalized_text(question_problem).lower() not in BOOLEAN_VALUES:
            raise ValueError(f"Invalid adjudicated question/key value for {unit_id}")
        result[unit_id] = {
            "primary_label": primary,
            "secondary_label": secondary,
            "question_or_key_problem": parse_boolean(question_problem, "adjudicated_question_or_key_problem", unit_id)
            if question_problem
            else None,
        }
    missing_critical = {
        unit_id
        for unit_id in eligible_ids
        if (
            rater_1[unit_id]["primary_label"] != rater_2[unit_id]["primary_label"]
            or rater_1[unit_id]["question_or_key_problem"] != rater_2[unit_id]["question_or_key_problem"]
        )
        and unit_id not in result
    }
    if missing_critical:
        raise ValueError(f"Adjudication CSV omits required decisions for {len(missing_critical)} disagreements")
    return result


def wilson_interval(proportion: float, n: float, z: float = 1.959963984540054) -> tuple[float | None, float | None]:
    if n <= 0:
        return None, None
    denominator = 1 + z * z / n
    centre = (proportion + z * z / (2 * n)) / denominator
    radius = z * math.sqrt((proportion * (1 - proportion) + z * z / (4 * n)) / n) / denominator
    return max(0.0, centre - radius), min(1.0, centre + radius)


def frequency_table(records: list[dict[str, Any]], multi_label: bool = False) -> dict[str, Any]:
    n = len(records)
    weights = [float(record["analysis_weight"]) for record in records]
    weight_total = sum(weights)
    effective_n = (weight_total * weight_total / sum(weight * weight for weight in weights)) if weights else 0.0
    rows = []
    for label in LABELS:
        if multi_label:
            matched = [record for record in records if label in record["all_labels"]]
        else:
            matched = [record for record in records if record["primary_label"] == label]
        count = len(matched)
        unweighted = count / n if n else None
        weighted_numerator = sum(float(record["analysis_weight"]) for record in matched)
        weighted = weighted_numerator / weight_total if weight_total else None
        unweighted_ci = wilson_interval(unweighted, n) if unweighted is not None else (None, None)
        weighted_ci = wilson_interval(weighted, effective_n) if weighted is not None else (None, None)
        rows.append(
            {
                "label": label,
                "unweighted_count": count,
                "unweighted_proportion": unweighted,
                "unweighted_wilson_95_ci": list(unweighted_ci),
                "weighted_total": weighted_numerator,
                "weighted_proportion": weighted,
                "weighted_wilson_95_ci_kish_effective_n": list(weighted_ci),
            }
        )
    return {"n": n, "weight_total": weight_total, "kish_effective_n": effective_n, "frequencies": rows}


def stratified_tables(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    def group(key: str) -> dict[str, list[dict[str, Any]]]:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            if key == "category":
                value = f"{record['category_id']}: {record['category']}"
            elif key == "control_pass":
                value = "pass" if record["control_pass"] else "fail"
            else:
                value = str(record[key])
            grouped[value].append(record)
        return dict(sorted(grouped.items()))

    return {dimension: {name: frequency_table(group_rows) for name, group_rows in group(dimension).items()} for dimension in ("model", "category", "control_pass")}


def finalize_records(
    audit: dict[str, dict[str, Any]],
    rater_1: dict[str, dict[str, Any]],
    rater_2: dict[str, dict[str, Any]],
    adjudications: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    records = []
    for unit_id in sorted(audit):
        one, two, adjudicated = rater_1[unit_id], rater_2[unit_id], adjudications.get(unit_id, {})
        if one["primary_label"] == two["primary_label"]:
            primary = one["primary_label"]
        else:
            primary = adjudicated["primary_label"]
        if one["question_or_key_problem"] == two["question_or_key_problem"]:
            question_problem = one["question_or_key_problem"]
        else:
            question_problem = adjudicated["question_or_key_problem"]
        secondary = adjudicated.get("secondary_label", "") or ""
        if not secondary:
            # This is explicitly a sensitivity definition, not a third inferred label.
            secondary_candidates = {one["secondary_label"], two["secondary_label"]} - {"", primary}
        else:
            secondary_candidates = {secondary}
        row = dict(audit[unit_id])
        row.update(
            primary_label=primary,
            secondary_labels=sorted(secondary_candidates),
            all_labels=sorted({primary, *secondary_candidates}),
            question_or_key_problem=question_problem,
            rater_confidence={"rater_1": one["confidence"], "rater_2": two["confidence"]},
            both_raters_nonlow_confidence=one["confidence"] != "low" and two["confidence"] != "low",
            rater_evidence={"rater_1": one["evidence_note"], "rater_2": two["evidence_note"]},
        )
        records.append(row)
    return records


def reliability_summary(rater_1: dict[str, dict[str, Any]], rater_2: dict[str, dict[str, Any]]) -> dict[str, Any]:
    ids = sorted(rater_1)
    primary_1 = [rater_1[unit_id]["primary_label"] for unit_id in ids]
    primary_2 = [rater_2[unit_id]["primary_label"] for unit_id in ids]
    problem_1 = [rater_1[unit_id]["question_or_key_problem"] for unit_id in ids]
    problem_2 = [rater_2[unit_id]["question_or_key_problem"] for unit_id in ids]
    secondary_agreement = sum(rater_1[unit_id]["secondary_label"] == rater_2[unit_id]["secondary_label"] for unit_id in ids)
    confusion = {
        first_label: {
            second_label: sum(
                rater_1[unit_id]["primary_label"] == first_label
                and rater_2[unit_id]["primary_label"] == second_label
                for unit_id in ids
            )
            for second_label in LABELS
        }
        for first_label in LABELS
    }
    per_label = {}
    for label in LABELS:
        both = sum(a == label and b == label for a, b in zip(primary_1, primary_2))
        either = sum(a == label or b == label for a, b in zip(primary_1, primary_2))
        per_label[label] = {
            "both_raters": both,
            "either_rater": either,
            "positive_agreement_jaccard": both / either if either else None,
        }
    return {
        "n": len(ids),
        "primary_label": {
            "raw_agreement": sum(a == b for a, b in zip(primary_1, primary_2)) / len(ids),
            "cohen_kappa": cohen_kappa(primary_1, primary_2),
            "rater_1_prevalence": label_prevalence(rater_1.values()),
            "rater_2_prevalence": label_prevalence(rater_2.values()),
            "confusion_matrix_rater_1_rows_rater_2_columns": confusion,
            "per_label_positive_agreement": per_label,
        },
        "question_or_key_problem": {
            "raw_agreement": sum(a == b for a, b in zip(problem_1, problem_2)) / len(ids),
            "rater_1_yes": sum(problem_1),
            "rater_2_yes": sum(problem_2),
        },
        "secondary_label_exact_agreement": secondary_agreement / len(ids),
    }


def format_percent(value: float | None) -> str:
    return "NA" if value is None else f"{100 * value:.1f}%"


def markdown_report(report: dict[str, Any]) -> str:
    reliability = report["reliability"]
    kappa = reliability["primary_label"]["cohen_kappa"]
    lines = [
        "# Failure-mode Annotation Analysis",
        "",
        "## Reliability",
        "",
        f"- Analysis units: {reliability['n']} model-item errors.",
        f"- Primary-label raw agreement: {format_percent(reliability['primary_label']['raw_agreement'])}.",
        f"- Cohen's kappa: {kappa['kappa'] if kappa['kappa'] is not None else 'undefined'} "
        f"(expected agreement {format_percent(kappa['expected_agreement'])}).",
        f"- Question/key-problem raw agreement: {format_percent(reliability['question_or_key_problem']['raw_agreement'])}.",
        f"- Secondary-label exact agreement: {format_percent(reliability['secondary_label_exact_agreement'])}.",
        "",
        "Kappa is reported alongside rater-specific label prevalence in the JSON output because rare labels can depress kappa despite high raw agreement.",
        "",
    ]
    if report["status"] != "adjudicated":
        lines += [
            "## Adjudication pending",
            "",
            f"The blinded queue contains {report['adjudication_queue_rows']} units with at least one disagreement. Final category frequencies are intentionally not produced until the required third-party adjudications are supplied.",
        ]
        return "\n".join(lines) + "\n"

    overall = report["final_statistics"]["overall_primary"]
    lines += [
        "## Adjudicated Primary-label Frequencies",
        "",
        "| Label | Unweighted n (%) | Weighted % | Unweighted Wilson 95% CI | Weighted Wilson 95% CI* |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in overall["frequencies"]:
        unweighted_ci = row["unweighted_wilson_95_ci"]
        weighted_ci = row["weighted_wilson_95_ci_kish_effective_n"]
        lines.append(
            f"| {row['label']} | {row['unweighted_count']} ({format_percent(row['unweighted_proportion'])}) | "
            f"{format_percent(row['weighted_proportion'])} | {format_percent(unweighted_ci[0])} to {format_percent(unweighted_ci[1])} | "
            f"{format_percent(weighted_ci[0])} to {format_percent(weighted_ci[1])} |"
        )
    lines += [
        "",
        f"*Weighted point estimates use inverse inclusion-probability weights. The weighted Wilson interval uses Kish effective n={overall['kish_effective_n']:.1f}; it is descriptive rather than a design-based confidence interval.",
        "",
        "## Scope and Inference",
        "",
        "The sampling and reporting unit is a model-item error. The 300 units map to "
        f"{report['cluster_diagnostic']['unique_benchmark_items']} benchmark items, and "
        f"{report['cluster_diagnostic']['benchmark_items_with_multiple_models']} items occur for more than one model. "
        "Those repeated items create within-item dependence. Therefore Wilson intervals are descriptive summaries of this stratified error sample, not independent-unit inferential tests or model-comparison p-values.",
        "",
        "The JSON output additionally contains model, six-category, and control-pass strata; a multi-label secondary-label sensitivity analysis; and exclusions for benchmark ambiguity alone and for benchmark ambiguity plus protocol/parsing cases.",
    ]
    return "\n".join(lines) + "\n"


def run_analysis(
    rater_1_path: Path,
    rater_2_path: Path,
    audit_manifest: Path,
    output_dir: Path,
    adjudicated_path: Path | None = None,
    expected_rows: int = 300,
) -> dict[str, Any]:
    """Run reliability-only or full post-adjudication failure-mode analysis."""
    audit = load_audit(audit_manifest, expected_rows)
    rater_1 = validate_rater_rows(read_csv(rater_1_path), audit, "rater_1", expected_rows)
    rater_2 = validate_rater_rows(read_csv(rater_2_path), audit, "rater_2", expected_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    queue_path = output_dir / "adjudication_queue.csv"
    if adjudicated_path is not None and adjudicated_path.resolve() == queue_path.resolve():
        raise ValueError(
            "The completed adjudication input must be a separate file from the generated "
            "adjudication_queue.csv; copy it to adjudication_final.csv before final analysis"
        )
    queue_ids = build_disagreement_queue(rater_1, rater_2, queue_path)
    report: dict[str, Any] = {
        "schema_version": "error-failure-mode-analysis-v1",
        "status": "adjudicated" if adjudicated_path else "awaiting_adjudication",
        "input": {
            "audit_manifest": str(audit_manifest),
            "rater_1": str(rater_1_path),
            "rater_2": str(rater_2_path),
            "adjudicated": str(adjudicated_path) if adjudicated_path else None,
        },
        "validation": {
            "expected_rows": expected_rows,
            "audit_rows": len(audit),
            "rater_rows": {"rater_1": len(rater_1), "rater_2": len(rater_2)},
            "unit_ids_exactly_aligned": True,
            "immutable_context_matches_audit": True,
            "allowed_primary_labels": list(LABELS),
        },
        "reliability": reliability_summary(rater_1, rater_2),
        "adjudication_queue": str(queue_path),
        "adjudication_queue_rows": len(queue_ids),
    }
    if adjudicated_path:
        adjudications = load_adjudications(adjudicated_path, set(queue_ids), rater_1, rater_2)
        records = finalize_records(audit, rater_1, rater_2, adjudications)
        benchmark_counts = Counter(record["benchmark_index"] for record in records)
        excluded_ambiguity = [row for row in records if row["primary_label"] != "benchmark_ambiguity"]
        excluded_noncontent = [
            row
            for row in records
            if row["primary_label"] not in {"benchmark_ambiguity", "protocol_or_parsing"}
        ]
        nonlow_confidence = [row for row in records if row["both_raters_nonlow_confidence"]]
        representative_evidence = {}
        for label in LABELS:
            candidates = [
                row
                for row in records
                if row["primary_label"] == label and row["both_raters_nonlow_confidence"]
            ]
            candidates.sort(key=lambda row: (-float(row["analysis_weight"]), row["unit_id"]))
            representative_evidence[label] = [
                {
                    "unit_id": row["unit_id"],
                    "benchmark_index": row["benchmark_index"],
                    "category_id": row["category_id"],
                    "control_pass": row["control_pass"],
                    "question": row["question"],
                    "gold_answer": row["gold_answer"],
                    "model_answer": row["model_answer"],
                    "rater_evidence": row["rater_evidence"],
                }
                for row in candidates[:3]
            ]
        report["cluster_diagnostic"] = {
            "analysis_units": len(records),
            "unique_benchmark_items": len(benchmark_counts),
            "benchmark_items_with_multiple_models": sum(count > 1 for count in benchmark_counts.values()),
            "max_models_per_benchmark_item": max(benchmark_counts.values()),
            "inference_unit_limitation": "Model-item errors sharing a benchmark item are correlated; reported Wilson intervals are descriptive and are not treated as independent-unit inference.",
        }
        report["final_statistics"] = {
            "overall_primary": frequency_table(records),
            "stratified_primary": stratified_tables(records),
            "secondary_multilabel_sensitivity": frequency_table(records, multi_label=True),
            "sensitivity_excluding_benchmark_ambiguity": frequency_table(excluded_ambiguity),
            "sensitivity_excluding_benchmark_ambiguity_and_protocol_or_parsing": frequency_table(excluded_noncontent),
            "sensitivity_both_raters_nonlow_confidence": frequency_table(nonlow_confidence),
            "confidence": {
                "rater_1": dict(Counter(row["rater_confidence"]["rater_1"] for row in records)),
                "rater_2": dict(Counter(row["rater_confidence"]["rater_2"] for row in records)),
                "both_raters_nonlow_count": len(nonlow_confidence),
                "both_raters_nonlow_proportion": len(nonlow_confidence) / len(records),
            },
            "question_or_key_problem": {
                "count": sum(row["question_or_key_problem"] for row in records),
                "proportion": sum(row["question_or_key_problem"] for row in records) / len(records),
                "wilson_95_ci": list(wilson_interval(sum(row["question_or_key_problem"] for row in records) / len(records), len(records))),
            },
            "representative_high_or_medium_confidence_evidence": representative_evidence,
        }
    json_path = output_dir / "analysis.json"
    markdown_path = output_dir / "analysis.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(markdown_report(report), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rater-1", type=Path, required=True)
    parser.add_argument("--rater-2", type=Path, required=True)
    parser.add_argument("--audit-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--adjudicated", type=Path, help="Completed adjudication_queue.csv; omit for reliability-only stage")
    parser.add_argument("--expected-rows", type=int, default=300)
    args = parser.parse_args()
    report = run_analysis(
        args.rater_1,
        args.rater_2,
        args.audit_manifest,
        args.output_dir,
        args.adjudicated,
        args.expected_rows,
    )
    print(json.dumps({"status": report["status"], "adjudication_queue_rows": report["adjudication_queue_rows"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
