#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path
from typing import Any, Sequence

from generate_probe_drafts import (
    append_usage,
    encode_image,
    parse_openai_chat_response,
    parse_openai_responses_response,
    post_json,
    read_usage_totals,
    strip_code_fence,
)


MODEL = "Qwen/Qwen3-VL-235B-A22B-Instruct"
FIELDNAMES = (
    "probe_sample_id",
    "source_sample_id",
    "benchmark_index",
    "video",
    "probe_type",
    "probe_subtype",
    "question",
    "option_a",
    "option_b",
    "option_c",
    "option_d",
    "answer",
    "evidence_description",
    "draft_rationale",
    "validation_status",
    "annotator_id",
    "reviewer_1_id",
    "reviewer_1_status",
    "reviewer_2_id",
    "reviewer_2_status",
    "adjudication_status",
    "notes",
)

SYSTEM_PROMPT = """You draft one prerequisite visual-perception control for a
scientific video benchmark. The supplied image is a contact sheet sampled
uniformly across the full source video.

The assigned probe type and subtype are fixed. Write exactly one English
four-choice question that tests a directly visible prerequisite for the
original benchmark question. The control must be simpler than the original.
It must not ask about intent, goals, causality, prediction, counterfactuals,
hidden state, exact OCR, or details too small to verify. Do not restate or
answer the original question. Do not mention contact sheets, frames,
timestamps, timecodes, or numeric time intervals in the question or choices.

The original question may explicitly state visible facts. Never test a fact
already stated in its wording: a reader who sees only the original question
must not be able to answer the control. Instead test a different, directly
visible prerequisite needed to interpret the original question.

Example: if the original says "the rider points with her left hand", asking
which hand points is invalid. A valid action control may instead ask what the
rider and horse are doing before the gesture, provided that action is visible
and not stated in the original. Before drafting, identify every visible fact
given away by the original question and exclude all of them as answer targets.

The correct answer and all distractors must be concise, mutually exclusive,
and visually plausible, with exactly one answer supported. Keep each choice at
12 words or fewer and at the same semantic granularity. For action_recognition,
ask about one atomic visible action at a concrete part of the event; do not ask
what happens "throughout", "overall", or "in a continuous path". For
temporal_order, compare two named visible events using before or after. Return
only JSON:
{
  "probe_type": "assigned type",
  "probe_subtype": "assigned subtype",
  "question": "...",
  "correct_answer": "...",
  "distractors": ["...", "...", "..."],
  "evidence_description": "what visible evidence establishes the answer",
  "draft_rationale": "why this is a prerequisite for the original question",
  "original_facts_avoided": ["visible facts stated by the original question"],
  "answerable_from_original_text_only": false
}
"""


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_attempted_sample_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    attempted = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("status") not in {"ok", "content_rejected"}:
            continue
        sample_id = str(record.get("probe_sample_id", "")).strip()
        if sample_id:
            attempted.add(sample_id)
    return attempted


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def parse_payload(text: str, row: dict[str, str]) -> dict[str, Any]:
    payload = json.loads(strip_code_fence(text))
    if not isinstance(payload, dict):
        raise ValueError("response must be a JSON object")
    for key in (
        "probe_type",
        "probe_subtype",
        "question",
        "correct_answer",
        "evidence_description",
        "draft_rationale",
        "original_facts_avoided",
    ):
        if not str(payload.get(key, "")).strip():
            raise ValueError(f"missing {key}")
    if payload.get("answerable_from_original_text_only") is not False:
        raise ValueError("model did not certify that the answer requires visual evidence")
    if not isinstance(payload["original_facts_avoided"], list):
        raise ValueError("original_facts_avoided must be a list")
    if payload["probe_type"] != row["assigned_probe_type"]:
        raise ValueError("model changed the assigned probe type")
    if payload["probe_subtype"] != row["assigned_probe_subtype"]:
        raise ValueError("model changed the assigned probe subtype")
    distractors = payload.get("distractors")
    if not isinstance(distractors, list) or len(distractors) != 3:
        raise ValueError("exactly three distractors are required")
    distractors = [str(value).strip() for value in distractors]
    correct = str(payload["correct_answer"]).strip()
    if any(not value for value in distractors):
        raise ValueError("distractors cannot be blank")
    if len({value.casefold() for value in [correct, *distractors]}) != 4:
        raise ValueError("choices must be distinct")
    if any(len(re.findall(r"\b[\w'-]+\b", value)) > 12 for value in [correct, *distractors]):
        raise ValueError("choices must contain no more than 12 words")

    normalized_original = re.sub(
        r"[^a-z0-9]+", " ", row["original_question"].casefold()
    ).strip()
    normalized_correct = re.sub(r"[^a-z0-9]+", " ", correct.casefold()).strip()
    if len(normalized_correct) >= 4 and re.search(
        rf"(?<![a-z0-9]){re.escape(normalized_correct)}(?![a-z0-9])",
        normalized_original,
    ):
        raise ValueError("correct answer is explicitly stated in the original question")

    rendered = " ".join([str(payload["question"]), correct, *distractors])
    if re.search(
        r"\b(contact sheet|frames?|timestamps?|timecodes?)\b"
        r"|\b\d{1,2}:\d{2}(?::\d{2})?(?:\.\d+)?\b",
        rendered,
        re.IGNORECASE,
    ):
        raise ValueError("question or choices contain forbidden presentation language")
    if re.search(
        r"\b(intent|intend|goal|purpose|why|predict|prediction|would|could have|"
        r"counterfactual|most likely)\b",
        str(payload["question"]),
        re.IGNORECASE,
    ):
        raise ValueError("question asks for higher-level reasoning")
    if row["assigned_probe_subtype"] == "action_recognition" and re.search(
        r"\b(throughout|overall|continuous(?:ly)?|mainly)\b",
        str(payload["question"]),
        re.IGNORECASE,
    ):
        raise ValueError("action_recognition question is not temporally atomic")
    if row["assigned_probe_subtype"] == "temporal_order" and not re.search(
        r"\b(before|after)\b", str(payload["question"]), re.IGNORECASE
    ):
        raise ValueError("temporal_order question must use before or after")
    payload["distractors"] = distractors
    return payload


def build_output_row(row: dict[str, str], payload: dict[str, Any], model: str) -> dict[str, str]:
    target = row["target_answer_position"]
    position = "ABCD".index(target)
    choices = list(payload["distractors"])
    choices.insert(position, str(payload["correct_answer"]).strip())
    return {
        "probe_sample_id": row["probe_sample_id"],
        "source_sample_id": row["source_sample_id"],
        "benchmark_index": row["benchmark_index"],
        "video": row["video"],
        "probe_type": row["assigned_probe_type"],
        "probe_subtype": row["assigned_probe_subtype"],
        "question": str(payload["question"]).strip(),
        "option_a": choices[0],
        "option_b": choices[1],
        "option_c": choices[2],
        "option_d": choices[3],
        "answer": target,
        "evidence_description": str(payload["evidence_description"]).strip(),
        "draft_rationale": str(payload["draft_rationale"]).strip(),
        "validation_status": "draft",
        "annotator_id": f"llm:{model}",
        "reviewer_1_id": "",
        "reviewer_1_status": "",
        "reviewer_2_id": "",
        "reviewer_2_status": "",
        "adjudication_status": "",
        "notes": "LLM-assisted draft; full-video independent review is required.",
    }


def call_model(
    api_key: str,
    base_url: str,
    model: str,
    row: dict[str, str],
    image_path: Path,
    max_output_tokens: int,
    timeout: float,
    protocol: str,
) -> dict[str, Any]:
    subtype_reminder = ""
    if row["assigned_probe_subtype"] == "temporal_order":
        subtype_reminder = (
            "\nMandatory retry constraint: the question sentence itself must literally "
            "contain the word 'before' or 'after'. Prefer the form 'What happens before X?' "
            "or 'What happens after X?'."
        )
    user_text = (
        f"Assigned probe type: {row['assigned_probe_type']}\n"
        f"Assigned probe subtype: {row['assigned_probe_subtype']}\n"
        "Original benchmark question (context only; do not answer or restate it):\n"
        f"{row['original_question']}\n"
        "Mandatory anti-leakage constraint: do not copy any complete answer phrase from "
        "the original question into the correct answer. Test a different visible fact."
        f"{subtype_reminder}"
    )
    encoded_image = encode_image(image_path)
    authorization = {"Authorization": f"Bearer {api_key}"}
    if protocol == "openai-chat":
        response = post_json(
            f"{base_url.rstrip('/')}/chat/completions",
            {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/jpeg;base64," + encoded_image
                            },
                        },
                    ],
                },
            ],
            "max_tokens": max_output_tokens,
            "temperature": 0.0,
            "stream": False,
            },
            authorization,
            timeout,
        )
        return parse_openai_chat_response(response)
    if protocol == "openai-responses":
        response = post_json(
            f"{base_url.rstrip('/')}/responses",
            {
                "model": model,
                "instructions": SYSTEM_PROMPT,
                "input": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": user_text},
                            {
                                "type": "input_image",
                                "image_url": "data:image/jpeg;base64," + encoded_image,
                            },
                        ],
                    }
                ],
                "max_output_tokens": max_output_tokens,
            },
            authorization,
            timeout,
        )
        return parse_openai_responses_response(response)
    raise ValueError(f"unsupported protocol: {protocol}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draft one assigned perception-control probe per video.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--usage-log", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument(
        "--protocol",
        choices=("openai-chat", "openai-responses"),
        default="openai-chat",
    )
    parser.add_argument("--base-url", default="https://api.umodelverse.ai/v1")
    parser.add_argument("--api-key-env", default="UMODELVERSE_API_KEY")
    parser.add_argument("--sample-id", action="append", default=[])
    parser.add_argument("--max-samples", type=int, default=20)
    parser.add_argument("--max-output-tokens", type=int, default=700)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--input-cny-per-million", type=float, default=2.0)
    parser.add_argument("--output-cny-per-million", type=float, default=8.0)
    parser.add_argument("--estimated-input-tokens", type=int, default=8_000)
    parser.add_argument("--max-estimated-cost-cny", type=float, default=1.0)
    parser.add_argument("--continue-on-rejection", action="store_true")
    parser.add_argument("--retry-attempted", action="store_true")
    parser.add_argument("--live", action="store_true")
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    manifest = read_csv(args.manifest)
    requested = set(args.sample_id)
    existing = read_csv(args.output) if args.output.exists() else []
    completed = {row["probe_sample_id"] for row in existing}
    attempted = read_attempted_sample_ids(args.usage_log)
    excluded = completed if args.retry_attempted else completed | attempted
    selected = [
        row
        for row in manifest
        if row["probe_sample_id"] not in excluded
        and (not requested or row["probe_sample_id"] in requested)
        and (args.image_dir / f"{row['source_sample_id']}.jpg").is_file()
    ][: args.max_samples]
    estimated_cost = len(selected) * (
        args.estimated_input_tokens * args.input_cny_per_million
        + args.max_output_tokens * args.output_cny_per_million
    ) / 1_000_000
    prior_requests, prior_input, prior_output = read_usage_totals(args.usage_log)
    prior_cost = (
        prior_input * args.input_cny_per_million
        + prior_output * args.output_cny_per_million
    ) / 1_000_000
    projected_total_cost = prior_cost + estimated_cost
    print(
        json.dumps(
            {
                "live": args.live,
                "model": args.model,
                "protocol": args.protocol,
                "selected": [row["probe_sample_id"] for row in selected],
                "requests": len(selected),
                "estimated_cost_cny": round(estimated_cost, 4),
                "prior_estimated_cost_cny": round(prior_cost, 6),
                "projected_total_cost_cny": round(projected_total_cost, 4),
                "cumulative_cost_cap_cny": args.max_estimated_cost_cny,
                "prior_billable_requests": prior_requests,
                "prior_tokens": prior_input + prior_output,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if projected_total_cost > args.max_estimated_cost_cny:
        raise SystemExit("projected cumulative cost exceeds cap")
    if not args.live or not selected:
        return 0
    api_key = os.getenv(args.api_key_env)
    if not api_key:
        raise SystemExit(f"missing {args.api_key_env}")

    output_rows = list(existing)
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    for row in selected:
        usage: dict[str, Any] | None = None
        try:
            response = call_model(
                api_key,
                args.base_url,
                args.model,
                row,
                args.image_dir / f"{row['source_sample_id']}.jpg",
                args.max_output_tokens,
                args.timeout,
                args.protocol,
            )
            usage = {
                "probe_sample_id": row["probe_sample_id"],
                "source_sample_id": row["source_sample_id"],
                "model": args.model,
                "protocol": args.protocol,
                "request_id": response.get("request_id"),
                "status": "ok",
                "prompt_tokens": response.get("prompt_tokens"),
                "completion_tokens": response.get("completion_tokens"),
                "total_tokens": response.get("total_tokens"),
            }
            (args.raw_dir / f"{row['probe_sample_id']}.json").write_text(
                json.dumps({**usage, "content": response["content"]}, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            payload = parse_payload(response["content"], row)
            output_rows.append(build_output_row(row, payload, args.model))
            write_csv(args.output, output_rows)
            append_usage(args.usage_log, usage)
            print(f"{row['probe_sample_id']}: draft written")
        except Exception as error:
            rejected = {
                "probe_sample_id": row["probe_sample_id"],
                "source_sample_id": row["source_sample_id"],
                "model": args.model,
                "protocol": args.protocol,
                "status": "content_rejected" if usage else "api_error",
                "error_type": type(error).__name__,
                "error": str(error)[:500],
            }
            if usage:
                rejected = {**usage, **rejected}
            append_usage(args.usage_log, rejected)
            if args.continue_on_rejection:
                print(f"{row['probe_sample_id']}: rejected: {error}")
                continue
            raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
