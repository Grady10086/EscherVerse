#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_label_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Use LABEL=PATH")
    label, raw_path = value.split("=", 1)
    if not label.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("Use non-empty LABEL=PATH")
    return label.strip(), Path(raw_path)


def pct(value: Any) -> str:
    return "NA" if value is None else f"{100 * float(value):.1f}%"


def pp(value: Any) -> str:
    return "NA" if value is None else f"{100 * float(value):+.1f} pp"


def analysis_record(label: str, path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "label": label,
        "path": str(path),
        "sha256": sha256(path),
        "candidate_pairs_sha256": payload["candidate_pairs_sha256"],
        "coverage": payload["prediction_item_coverage"],
        "result": payload["subsets"]["all_candidates"],
        "reporting_guardrail": payload["reporting_guardrail"],
    }


def markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    return [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" if i == 0 else "---:" for i in range(len(headers))) + "|",
        *("| " + " | ".join(row) + " |" for row in rows),
    ]


def run(args: argparse.Namespace) -> dict[str, Any]:
    summary = json.loads(args.selection_summary.read_text(encoding="utf-8"))
    expected_candidate_sha256 = sha256(args.candidate_pairs)
    if expected_candidate_sha256 != args.expected_candidate_sha256:
        raise ValueError(
            "Frozen candidate hash mismatch: "
            f"{expected_candidate_sha256} != {args.expected_candidate_sha256}"
        )
    if summary["cardinalities"]["candidate_pairs"] != 50:
        raise ValueError("Selection summary is not a 50-pair manifest")
    if not summary["selection"]["post_hoc_outcome_selected"]:
        raise ValueError("Selection summary does not disclose post-hoc selection")

    selection_labels = {
        "qwen3vl8b": "Qwen3-VL-8B-Instruct",
        "qwen25vl7b": "Qwen2.5-VL-7B-Instruct",
        "qwen3vl2b": "Qwen3-VL-2B-Instruct",
        "internvl3_2b": "InternVL3-2B",
    }
    selection = []
    for key, result in summary["descriptive_selected_set_results"].items():
        selection.append(
            {
                "label": selection_labels.get(key, key),
                "key": key,
                **result,
            }
        )

    heldout = [analysis_record(label, path) for label, path in args.heldout]
    sensitivity = {
        label: analysis_record(label, path)
        for label, path in (args.sensitivity or [])
    }
    historical = [
        analysis_record(label, path) for label, path in (args.historical or [])
    ]
    for record in [*heldout, *sensitivity.values(), *historical]:
        if record["candidate_pairs_sha256"] != expected_candidate_sha256:
            raise ValueError(
                f"Candidate hash mismatch for {record['label']}: "
                f"{record['candidate_pairs_sha256']}"
            )
    for record in heldout:
        coverage = record["coverage"]
        if coverage["covered"] != coverage["required"]:
            raise ValueError(f"Held-out result is incomplete: {record['label']}")

    report = {
        "schema_version": "posthoc-natural50-model-panel-v1",
        "candidate_pairs_sha256": expected_candidate_sha256,
        "candidate_pairs": {
            "path": str(args.candidate_pairs),
            "sha256": expected_candidate_sha256,
        },
        "selection_summary": {
            "path": str(args.selection_summary),
            "sha256": sha256(args.selection_summary),
        },
        "selection_models": selection,
        "heldout_models": heldout,
        "sensitivity_analyses": sensitivity,
        "historical_partial_coverage": historical,
        "interpretation": {
            "selection_models": (
                "Descriptive only: these outcomes were used in post-hoc selection."
            ),
            "heldout_models": (
                "Evaluated only after the 50-pair set was frozen; no reselection."
            ),
            "scope": (
                "The panel validates behavior on a post-hoc same-video "
                "category-proxy set and applies only to these frozen 50 pairs."
            ),
        },
    }

    selection_rows = [
        [
            row["label"],
            f"{row['pairs']}/{row['pairs']}",
            pct(row["arm_a_accuracy"]),
            pct(row["arm_b_accuracy"]),
            pp(row["b_minus_a"]),
        ]
        for row in selection
    ]
    heldout_rows = []
    for row in heldout:
        result = row["result"]
        interval = result["video_cluster_bootstrap_interval"]
        sensitivity_delta = (
            f"{pp(result['paired_difference_b_minus_a'])}（无触顶）"
        )
        if row["label"] in sensitivity:
            sensitivity_delta = pp(
                sensitivity[row["label"]]["result"]["paired_difference_b_minus_a"]
            )
        heldout_rows.append(
            [
                row["label"],
                f"{result['n_pairs']}/{result['requested_pairs']}",
                pct(result["arm_a_accuracy"]),
                pct(result["arm_b_accuracy"]),
                pp(result["paired_difference_b_minus_a"]),
                f"[{pp(interval[0])}, {pp(interval[1])}]",
                f"{float(result['video_cluster_sign_flip_p']):.4f}",
                sensitivity_delta,
            ]
        )
    historical_rows = []
    for row in historical:
        result = row["result"]
        historical_rows.append(
            [
                row["label"],
                f"{row['coverage']['covered']}/{row['coverage']['required']}",
                f"{result['n_pairs']}/{result['requested_pairs']}",
                pct(result["arm_a_accuracy"]),
                pct(result["arm_b_accuracy"]),
                pp(result["paired_difference_b_minus_a"]),
            ]
        )

    lines = [
        "# 自然差值 50 对：模型结果面板",
        "",
        f"- 冻结候选 SHA-256：`{expected_candidate_sha256}`",
        "- Arm A/B 是同视频类别代理方向，不等同于逐对严格控制后的因果难度层级。",
        "",
        "## 参与事后选题的模型",
        "",
        "这些模型的逐题结果参与了50对的事后筛选，只能作描述，不能算独立验证。",
        "",
        *markdown_table(
            ["模型", "配对", "Arm A", "Arm B", "B-A"], selection_rows
        ),
        "",
        "## 冻结后新增模型",
        "",
        "候选集合冻结后才运行以下模型，未按结果重新选题。Thinking 模型主结果将所有最终仍达到 token 上限的回答保守判错；最后一列给出原始解析敏感性。32B-AWQ 无回答触及 token 上限，因此该列与主结果相同。",
        "",
        *markdown_table(
            [
                "模型",
                "配对",
                "Arm A",
                "Arm B",
                "B-A（主）",
                "95%视频聚类区间",
                "sign-flip p",
                "原始解析 B-A",
            ],
            heldout_rows,
        ),
    ]
    if historical_rows:
        lines.extend(
            [
                "",
                "## 历史部分覆盖诊断",
                "",
                "历史覆盖不是随机缺失，只作为补充诊断，不能与完整 held-out 结果等量合并。SenseNova-SI-1.3-InternVL3-8B 也不能改名为原生 OpenGVLab InternVL3-8B。",
                "",
                *markdown_table(
                    ["模型", "题目覆盖", "完整配对", "Arm A", "Arm B", "B-A"],
                    historical_rows,
                ),
            ]
        )
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "完整held-out模型可以检验该冻结50对上的方向是否迁移到未参与选题的模型，但不能消除50对本身为事后类别代理集合这一事实；结论只适用于这批冻结配对。",
            "",
        ]
    )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.output_md.write_text("\n".join(lines), encoding="utf-8")
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a disclosure-separated panel for the post-hoc natural 50 set."
    )
    parser.add_argument("--selection-summary", type=Path, required=True)
    parser.add_argument("--candidate-pairs", type=Path, required=True)
    parser.add_argument("--expected-candidate-sha256", required=True)
    parser.add_argument("--heldout", action="append", type=parse_label_path, required=True)
    parser.add_argument("--sensitivity", action="append", type=parse_label_path)
    parser.add_argument("--historical", action="append", type=parse_label_path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args(argv)


def main() -> int:
    report = run(parse_args())
    print(
        f"Wrote panel with {len(report['selection_models'])} selection and "
        f"{len(report['heldout_models'])} held-out models"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
