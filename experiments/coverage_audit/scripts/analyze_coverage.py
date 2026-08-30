#!/usr/bin/env python3
"""Analyze independent-author coverage judgments with video clustering."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path


CAPABILITIES = [
    "occlusion_permanence", "dynamic_relation", "action_goal",
    "physical_prediction", "deformation_state", "reference_frame",
    "orientation_facing", "temporal_order", "entity_attribute",
    "action_event", "static_relation", "other_askable",
]
AUTHORS = ["author_a", "author_b"]
SEED = 20260813


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def jsd(counts_a: Counter[str], counts_b: Counter[str]) -> float:
    total_a, total_b = sum(counts_a.values()), sum(counts_b.values())
    pa = [counts_a[key] / total_a for key in CAPABILITIES]
    pb = [counts_b[key] / total_b for key in CAPABILITIES]
    midpoint = [(a + b) / 2 for a, b in zip(pa, pb)]

    def kl(p: list[float], q: list[float]) -> float:
        return sum(x * math.log2(x / y) for x, y in zip(p, q) if x > 0)

    return (kl(pa, midpoint) + kl(pb, midpoint)) / 2


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def rate(rows: list[dict[str, str]], accepted: set[str]) -> tuple[float, int, int]:
    eligible = [row for row in rows if row["coverage_status"] != "uncertain"]
    numerator = sum(row["coverage_status"] in accepted for row in eligible)
    return numerator / len(eligible) if eligible else float("nan"), numerator, len(eligible)


def bootstrap(
    rows: list[dict[str, str]], videos: list[str], accepted: set[str], replicates: int
) -> tuple[float, float]:
    by_video: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_video[row["audit_item_id"]].append(row)
    rng = random.Random(SEED + len(accepted))
    estimates = []
    for _ in range(replicates):
        sampled = [rng.choice(videos) for _ in videos]
        replicate_rows = [row for video in sampled for row in by_video[video]]
        estimates.append(rate(replicate_rows, accepted)[0])
    return percentile(estimates, 0.025), percentile(estimates, 0.975)


def binary_kappa(pairs: list[tuple[int, int]]) -> float | None:
    observed = sum(a == b for a, b in pairs) / len(pairs)
    pa = sum(a for a, _ in pairs) / len(pairs)
    pb = sum(b for _, b in pairs) / len(pairs)
    expected = pa * pb + (1 - pa) * (1 - pb)
    if expected == 1:
        return None
    return (observed - expected) / (1 - expected)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--author-a", type=Path, required=True)
    parser.add_argument("--author-b", type=Path, required=True)
    parser.add_argument("--generated", type=Path, required=True)
    parser.add_argument("--judgments", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    args = parser.parse_args()

    authored = {"author_a": read_csv(args.author_a), "author_b": read_csv(args.author_b)}
    generated = read_csv(args.generated)
    judgments = read_csv(args.judgments)
    proposal_lookup = {
        (author, row["audit_item_id"], row["proposal_no"]): row
        for author, rows in authored.items() for row in rows
    }
    judgment_keys = {
        (row["author_id"], row["audit_item_id"], row["proposal_no"])
        for row in judgments
    }
    if set(proposal_lookup) != judgment_keys:
        raise ValueError("Coverage judgments do not exactly match authored proposals")
    if any(row["coverage_status"] not in {"direct", "partial", "absent", "uncertain"} for row in judgments):
        raise ValueError("Invalid coverage status")
    videos = sorted({row["audit_item_id"] for rows in authored.values() for row in rows})

    coverage = {}
    for author in AUTHORS + ["pooled"]:
        rows = judgments if author == "pooled" else [r for r in judgments if r["author_id"] == author]
        strict, sn, sd = rate(rows, {"direct"})
        lenient, ln, ld = rate(rows, {"direct", "partial"})
        strict_ci = bootstrap(rows, videos, {"direct"}, args.bootstrap)
        lenient_ci = bootstrap(rows, videos, {"direct", "partial"}, args.bootstrap)
        coverage[author] = {
            "status_counts": dict(sorted(Counter(r["coverage_status"] for r in rows).items())),
            "strict": {"rate": strict, "numerator": sn, "denominator": sd, "video_cluster_bootstrap_95ci": strict_ci},
            "lenient": {"rate": lenient, "numerator": ln, "denominator": ld, "video_cluster_bootstrap_95ci": lenient_ci},
        }

    generated_counts = Counter(row["primary_capability"] for row in generated)
    distributions = {}
    for author, rows in authored.items():
        counts = Counter(row["primary_capability"] for row in rows)
        distributions[author] = {
            "counts": dict(sorted(counts.items())),
            "jsd_vs_generated_bits": jsd(counts, generated_counts),
        }
    pooled_counts = Counter(row["primary_capability"] for rows in authored.values() for row in rows)
    distributions["pooled"] = {
        "counts": dict(sorted(pooled_counts.items())),
        "jsd_vs_generated_bits": jsd(pooled_counts, generated_counts),
    }

    sets: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for author, rows in authored.items():
        for row in rows:
            sets[author][row["audit_item_id"]].add(row["primary_capability"])
    jaccards = []
    binary_pairs = []
    for video in videos:
        a, b = sets["author_a"][video], sets["author_b"][video]
        jaccards.append(len(a & b) / len(a | b) if a | b else 1.0)
        for capability in CAPABILITIES:
            binary_pairs.append((int(capability in a), int(capability in b)))
    agreement = {
        "mean_video_capability_jaccard": sum(jaccards) / len(jaccards),
        "pooled_binary_raw_agreement": sum(a == b for a, b in binary_pairs) / len(binary_pairs),
        "pooled_binary_cohen_kappa": binary_kappa(binary_pairs),
    }

    by_capability = {}
    for capability in CAPABILITIES:
        rows = [
            judgment for judgment in judgments
            if proposal_lookup[(judgment["author_id"], judgment["audit_item_id"], judgment["proposal_no"])]["primary_capability"] == capability
        ]
        if rows:
            strict, sn, sd = rate(rows, {"direct"})
            lenient, ln, ld = rate(rows, {"direct", "partial"})
            by_capability[capability] = {
                "proposals": len(rows), "strict_rate": strict, "strict_n": sn,
                "lenient_rate": lenient, "lenient_n": ln, "denominator": sd,
            }

    report = {
        "schema": "escher-audit-coverage-analysis-v1",
        "videos": len(videos),
        "authored_proposals": sum(len(rows) for rows in authored.values()),
        "generated_same_video_questions": len(generated),
        "coverage": coverage,
        "generated_capability_counts": dict(sorted(generated_counts.items())),
        "author_distributions": distributions,
        "inter_author_agreement": agreement,
        "coverage_by_capability": by_capability,
        "bootstrap_replicates": args.bootstrap,
        "limitations": [
            "Both independent authors and the coverage judge are reviewer-assisted, not human-only raters.",
            "Authors primarily viewed 20-frame full-duration contact sheets rather than continuous full video.",
            "The generated-set capability labels are category-aligned deterministic mappings.",
            "This audit characterizes sampled coverage; it cannot prove exhaustive coverage.",
        ],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    pooled = coverage["pooled"]
    markdown = f"""# E5b 独立问题覆盖审计结果

## 主结果

- 视频：{len(videos)}；独立提议：{report['authored_proposals']}；同视频最终生成题：{len(generated)}。
- 严格覆盖率（direct）：{pooled['strict']['rate']:.1%}（{pooled['strict']['numerator']}/{pooled['strict']['denominator']}；视频聚类 bootstrap 95% CI {pooled['strict']['video_cluster_bootstrap_95ci'][0]:.1%}–{pooled['strict']['video_cluster_bootstrap_95ci'][1]:.1%}）。
- 宽松覆盖率（direct + partial）：{pooled['lenient']['rate']:.1%}（{pooled['lenient']['numerator']}/{pooled['lenient']['denominator']}；95% CI {pooled['lenient']['video_cluster_bootstrap_95ci'][0]:.1%}–{pooled['lenient']['video_cluster_bootstrap_95ci'][1]:.1%}）。
- 两位作者的视频级能力集合平均 Jaccard：{agreement['mean_video_capability_jaccard']:.3f}；二元能力存在性 pooled Cohen's kappa：{agreement['pooled_binary_cohen_kappa'] if agreement['pooled_binary_cohen_kappa'] is not None else 'NA'}。
- Pooled 独立参考集与最终生成题的能力分布 JSD：{distributions['pooled']['jsd_vs_generated_bits']:.3f} bits。

## 解释

严格指标只把询问同一核心事实且要求相同能力的题计为覆盖；宽松指标纳入部分重合。所有区间按视频而非逐提议重采样。

本轮两位独立作者和覆盖判定者均为先进多模态模型辅助角色，主要查看 20 帧全时长接触表，并非真人完整视频标注。它可以作为低成本 coverage characterization 和真人复核前的完整预实验，但不能在论文中写成 human-authored reference set。该审计也不证明不存在 coverage bias。
"""
    args.output_md.write_text(markdown, encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
