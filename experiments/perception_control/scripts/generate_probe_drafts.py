#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import re
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from common import OPTION_LETTERS, PROBE_SUBTYPES_BY_TYPE, PROBE_TYPES


PROBE_FIELDNAMES = (
    "sample_id",
    "benchmark_index",
    "video",
    "probe_id",
    "probe_type",
    "probe_subtype",
    "question",
    "option_a",
    "option_b",
    "option_c",
    "option_d",
    "answer",
    "validation_status",
    "annotator_id",
    "reviewer_id",
    "notes",
)

SYSTEM_PROMPT = """You draft prerequisite visual-perception controls for a
scientific video benchmark. The supplied image is a full-duration contact sheet
with uniformly sampled, timestamped frames from one video.

Return exactly three English four-choice probe drafts:
1. entity: directly identify a visible actor or object;
2. action_event: directly identify a visible action/event or simple event order;
3. simple_relation: directly identify a visible orientation or spatial relation.

The probes should test visual evidence needed to understand the original
benchmark question, but must not restate or answer that question. Do not ask
about intent, causality, prediction, counterfactuals, hidden state, exact OCR,
or details too small to verify. Every answer and distractor must be visually
plausible as a choice, while exactly one answer must be supported by the
contact sheet. Use only evidence visible in the supplied frames.

Write natural questions for someone watching the video. Never mention the
contact sheet, frames, timestamps, timecodes, or numeric time intervals. Do not
use text, brands, labels, or identifiers when a generic visual description is
enough. For action_recognition, ask what visibly happens. For temporal_order,
compare named visible events using first/before/after, never clock times.

Return one JSON object with this schema and no commentary:
{
  "probes": [
    {
      "probe_type": "entity",
      "probe_subtype": "object_actor_presence",
      "question": "...",
      "correct_answer": "...",
      "distractors": ["...", "...", "..."]
    },
    {
      "probe_type": "action_event",
      "probe_subtype": "action_recognition or temporal_order",
      "question": "...",
      "correct_answer": "...",
      "distractors": ["...", "...", "..."]
    },
    {
      "probe_type": "simple_relation",
      "probe_subtype": "orientation or simple_spatial_relation",
      "question": "...",
      "correct_answer": "...",
      "distractors": ["...", "...", "..."]
    }
  ]
}"""


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def strip_code_fence(text: str) -> str:
    cleaned = text.strip()
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", cleaned, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else cleaned


def parse_probe_payload(text: str) -> list[dict[str, Any]]:
    payload = json.loads(strip_code_fence(text))
    if not isinstance(payload, dict) or not isinstance(payload.get("probes"), list):
        raise ValueError("response must be an object containing a probes list")

    probes = payload["probes"]
    if len(probes) != len(PROBE_TYPES):
        raise ValueError(f"expected {len(PROBE_TYPES)} probes, found {len(probes)}")

    by_type: dict[str, dict[str, Any]] = {}
    for probe in probes:
        if not isinstance(probe, dict):
            raise ValueError("every probe must be a JSON object")
        probe_type = str(probe.get("probe_type", "")).strip()
        if probe_type not in PROBE_TYPES:
            raise ValueError(f"invalid probe_type {probe_type!r}")
        if probe_type in by_type:
            raise ValueError(f"duplicate probe_type {probe_type!r}")

        subtype = str(probe.get("probe_subtype", "")).strip()
        if subtype not in PROBE_SUBTYPES_BY_TYPE[probe_type]:
            raise ValueError(
                f"probe_subtype {subtype!r} is incompatible with {probe_type!r}"
            )
        question = str(probe.get("question", "")).strip()
        correct_answer = str(probe.get("correct_answer", "")).strip()
        distractors = probe.get("distractors")
        if not question or not correct_answer:
            raise ValueError(f"{probe_type}: question and correct_answer are required")
        if not isinstance(distractors, list) or len(distractors) != 3:
            raise ValueError(f"{probe_type}: exactly three distractors are required")
        distractors = [str(value).strip() for value in distractors]
        if any(not value for value in distractors):
            raise ValueError(f"{probe_type}: distractors cannot be empty")
        choices = [correct_answer, *distractors]
        if len({choice.casefold() for choice in choices}) != 4:
            raise ValueError(f"{probe_type}: answer choices must be distinct")

        by_type[probe_type] = {
            "probe_type": probe_type,
            "probe_subtype": subtype,
            "question": question,
            "correct_answer": correct_answer,
            "distractors": distractors,
        }
    return [by_type[probe_type] for probe_type in PROBE_TYPES]


def target_subtypes(sample_id: str) -> dict[str, str]:
    match = re.fullmatch(r"pc-(\d+)", sample_id)
    if not match:
        raise ValueError(f"unexpected sample_id format: {sample_id!r}")
    sample_index = int(match.group(1)) - 1
    return {
        "entity": "object_actor_presence",
        "action_event": (
            "action_recognition" if sample_index % 2 == 0 else "temporal_order"
        ),
        "simple_relation": (
            "orientation" if (sample_index // 2) % 2 == 0 else "simple_spatial_relation"
        ),
    }


def validate_probe_content(
    probes: list[dict[str, Any]],
    manifest_row: dict[str, str],
) -> None:
    expected_subtypes = target_subtypes(manifest_row["sample_id"])
    original_identifiers = {
        token.casefold()
        for token in re.findall(
            r"\b(?=[A-Za-z0-9'-]*[A-Za-z])(?=[A-Za-z0-9'-]*\d)"
            r"[A-Za-z0-9'-]{4,}\b",
            manifest_row["question"],
        )
    }
    forbidden_context = re.compile(
        r"\b(contact sheet|frames?|timestamps?|timecodes?|time (?:interval|window))\b"
        r"|\b\d{1,2}:\d{2}(?::\d{2})?(?:\.\d+)?\b",
        re.IGNORECASE,
    )
    for probe in probes:
        probe_type = probe["probe_type"]
        if probe["probe_subtype"] != expected_subtypes[probe_type]:
            raise ValueError(
                f"{probe_type}: expected subtype {expected_subtypes[probe_type]!r}, "
                f"found {probe['probe_subtype']!r}"
            )
        rendered = " ".join(
            [
                probe["question"],
                probe["correct_answer"],
                *probe["distractors"],
            ]
        )
        if forbidden_context.search(rendered):
            raise ValueError(f"{probe_type}: contains frame or timestamp language")
        rendered_tokens = {token.casefold() for token in re.findall(r"\b[\w'-]+\b", rendered)}
        repeated_identifiers = sorted(original_identifiers & rendered_tokens)
        if repeated_identifiers:
            raise ValueError(
                f"{probe_type}: repeats original alphanumeric identifier(s) "
                f"{repeated_identifiers}"
            )
        if probe["probe_subtype"] == "temporal_order" and not re.search(
            r"\b(first|before|after)\b", probe["question"], re.IGNORECASE
        ):
            raise ValueError(
                "temporal_order: question must compare named events using "
                "first, before, or after"
            )


def answer_position(sample_id: str, probe_index: int) -> int:
    match = re.fullmatch(r"pc-(\d+)", sample_id)
    if not match:
        raise ValueError(f"unexpected sample_id format: {sample_id!r}")
    sample_index = int(match.group(1)) - 1
    return (sample_index * len(PROBE_TYPES) + probe_index) % len(OPTION_LETTERS)


def build_probe_rows(
    manifest_row: dict[str, str],
    probes: list[dict[str, Any]],
    model: str,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    sample_id = manifest_row["sample_id"]
    for probe_index, probe in enumerate(probes):
        correct_position = answer_position(sample_id, probe_index)
        options = list(probe["distractors"])
        options.insert(correct_position, probe["correct_answer"])
        rows.append(
            {
                "sample_id": sample_id,
                "benchmark_index": manifest_row["benchmark_index"],
                "video": manifest_row["video"],
                "probe_id": f"{sample_id}-{probe['probe_type']}",
                "probe_type": probe["probe_type"],
                "probe_subtype": probe["probe_subtype"],
                "question": probe["question"],
                "option_a": options[0],
                "option_b": options[1],
                "option_c": options[2],
                "option_d": options[3],
                "answer": OPTION_LETTERS[correct_position],
                "validation_status": "draft",
                "annotator_id": f"llm:{model}",
                "reviewer_id": "",
                "notes": (
                    "LLM-assisted draft from a full-duration contact sheet; "
                    "independent full-video human review is required."
                ),
            }
        )
    return rows


def encode_image(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def estimate_cost_cny(
    requests: int,
    input_tokens_per_request: int,
    output_tokens_per_request: int,
    input_cny_per_million: float,
    output_cny_per_million: float,
) -> float:
    return requests * (
        input_tokens_per_request * input_cny_per_million
        + output_tokens_per_request * output_cny_per_million
    ) / 1_000_000


def read_usage_totals(path: Path) -> tuple[int, int, int]:
    if not path.exists():
        return 0, 0, 0
    requests = 0
    prompt_tokens = 0
    completion_tokens = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("status") not in {"ok", "content_rejected"}:
                continue
            requests += 1
            prompt_tokens += int(row.get("prompt_tokens") or 0)
            completion_tokens += int(row.get("completion_tokens") or 0)
    return requests, prompt_tokens, completion_tokens


def select_manifest_rows(
    manifest_rows: list[dict[str, str]],
    image_dir: Path,
    completed_sample_ids: set[str],
    requested_sample_ids: set[str],
    max_samples: int,
) -> list[dict[str, str]]:
    selected = []
    for row in manifest_rows:
        sample_id = row["sample_id"]
        if requested_sample_ids and sample_id not in requested_sample_ids:
            continue
        if sample_id in completed_sample_ids:
            continue
        if not (image_dir / f"{sample_id}.jpg").is_file():
            continue
        selected.append(row)
        if len(selected) >= max_samples:
            break
    return selected


def write_probe_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PROBE_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def append_usage(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def parse_openai_chat_response(response: dict[str, Any]) -> dict[str, Any]:
    usage = response.get("usage") or {}
    return {
        "request_id": response.get("id"),
        "content": response["choices"][0]["message"].get("content") or "",
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
    }


def parse_openai_responses_response(response: dict[str, Any]) -> dict[str, Any]:
    text_parts: list[str] = []
    if isinstance(response.get("output_text"), str):
        text_parts.append(response["output_text"])
    for item in response.get("output") or []:
        if not isinstance(item, dict):
            continue
        for part in item.get("content") or []:
            if not isinstance(part, dict):
                continue
            if part.get("type") in {"output_text", "text"} and isinstance(
                part.get("text"), str
            ):
                text_parts.append(part["text"])
    usage = response.get("usage") or {}
    return {
        "request_id": response.get("id"),
        "content": "\n".join(text_parts),
        "prompt_tokens": usage.get("input_tokens"),
        "completion_tokens": usage.get("output_tokens"),
        "total_tokens": usage.get("total_tokens"),
    }


def parse_gemini_response(response: dict[str, Any]) -> dict[str, Any]:
    text_parts: list[str] = []
    for candidate in response.get("candidates") or []:
        if not isinstance(candidate, dict):
            continue
        content = candidate.get("content") or {}
        for part in content.get("parts") or []:
            if (
                isinstance(part, dict)
                and not part.get("thought")
                and isinstance(part.get("text"), str)
            ):
                text_parts.append(part["text"])
    usage = response.get("usageMetadata") or {}
    return {
        "request_id": response.get("responseId"),
        "content": "\n".join(text_parts),
        "prompt_tokens": usage.get("promptTokenCount"),
        "completion_tokens": usage.get("candidatesTokenCount"),
        "total_tokens": usage.get("totalTokenCount"),
    }


def post_json(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout: float,
) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def call_model(
    api_key: str,
    base_url: str,
    model: str,
    protocol: str,
    manifest_row: dict[str, str],
    image_path: Path,
    max_output_tokens: int,
    timeout: float,
    gemini_thinking_budget: int,
) -> dict[str, Any]:
    user_text = (
        f"Sample ID: {manifest_row['sample_id']}\n"
        f"Required subtypes: {json.dumps(target_subtypes(manifest_row['sample_id']))}\n"
        f"Original benchmark question (context only; do not answer it):\n"
        f"{manifest_row['question']}"
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
                                    "url": (
                                        "data:image/jpeg;base64,"
                                        f"{encoded_image}"
                                    )
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
                                "image_url": (
                                    "data:image/jpeg;base64,"
                                    f"{encoded_image}"
                                ),
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

    if protocol == "gemini":
        api_root = re.sub(r"/v1/?$", "", base_url.rstrip("/"))
        quoted_model = urllib.parse.quote(model, safe="")
        response = post_json(
            f"{api_root}/v1beta/models/{quoted_model}:generateContent",
            {
                "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
                "contents": [
                    {
                        "role": "user",
                        "parts": [
                            {"text": user_text},
                            {
                                "inlineData": {
                                    "mimeType": "image/jpeg",
                                    "data": encoded_image,
                                }
                            },
                        ],
                    }
                ],
                "generationConfig": {
                    "temperature": 0.0,
                    "maxOutputTokens": max_output_tokens,
                    "responseMimeType": "application/json",
                    "thinkingConfig": {
                        "thinkingBudget": gemini_thinking_budget,
                    },
                },
            },
            authorization,
            timeout,
        )
        return parse_gemini_response(response)

    raise ValueError(f"unsupported protocol: {protocol}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draft perception probes with a cost-capped multimodal API call."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--usage-log", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--model", default="qwen3-vl-flash")
    parser.add_argument(
        "--protocol",
        choices=("openai-chat", "openai-responses", "gemini"),
        default="openai-chat",
    )
    parser.add_argument("--base-url", default="https://api.umodelverse.ai/v1/")
    parser.add_argument("--api-key-env", default="UMODELVERSE_API_KEY")
    parser.add_argument("--sample-id", action="append", default=[])
    parser.add_argument("--max-samples", type=int, default=5)
    parser.add_argument("--max-output-tokens", type=int, default=900)
    parser.add_argument("--estimated-input-tokens", type=int, default=32_000)
    parser.add_argument("--input-cny-per-million", type=float, default=0.15)
    parser.add_argument("--output-cny-per-million", type=float, default=1.5)
    parser.add_argument("--max-estimated-cost-cny", type=float, default=0.05)
    parser.add_argument("--max-cumulative-cost-cny", type=float, default=0.05)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--gemini-thinking-budget", type=int, default=0)
    parser.add_argument(
        "--live",
        action="store_true",
        help="Send API requests. Without this flag the script only prints its plan.",
    )
    parser.add_argument(
        "--continue-on-rejection",
        action="store_true",
        help=(
            "Continue to the next sample when a billable response fails content "
            "validation. API/transport errors still stop the run."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_samples < 1:
        raise SystemExit("--max-samples must be positive")
    if args.max_output_tokens < 1 or args.estimated_input_tokens < 1:
        raise SystemExit("token limits must be positive")

    manifest_rows = read_csv(args.manifest)
    existing_rows = read_csv(args.output) if args.output.exists() else []
    completed_sample_ids = {row["sample_id"] for row in existing_rows}
    requested_sample_ids = set(args.sample_id)
    selected = select_manifest_rows(
        manifest_rows,
        args.image_dir,
        completed_sample_ids,
        requested_sample_ids,
        args.max_samples,
    )
    estimated_cost = estimate_cost_cny(
        len(selected),
        args.estimated_input_tokens,
        args.max_output_tokens,
        args.input_cny_per_million,
        args.output_cny_per_million,
    )
    prior_requests, prior_prompt_tokens, prior_completion_tokens = read_usage_totals(
        args.usage_log
    )
    prior_cost = (
        prior_prompt_tokens * args.input_cny_per_million
        + prior_completion_tokens * args.output_cny_per_million
    ) / 1_000_000
    plan = {
        "live": args.live,
        "model": args.model,
        "protocol": args.protocol,
        "selected_samples": [row["sample_id"] for row in selected],
        "request_count": len(selected),
        "maximum_output_tokens": len(selected) * args.max_output_tokens,
        "estimated_cost_cny": round(estimated_cost, 6),
        "cost_cap_cny": args.max_estimated_cost_cny,
        "prior_billable_requests": prior_requests,
        "prior_estimated_actual_cost_cny": round(prior_cost, 6),
        "cumulative_estimated_upper_bound_cny": round(prior_cost + estimated_cost, 6),
        "cumulative_cost_cap_cny": args.max_cumulative_cost_cny,
        "existing_probe_rows": len(existing_rows),
    }
    print(json.dumps(plan, ensure_ascii=False, indent=2))

    if estimated_cost > args.max_estimated_cost_cny:
        raise SystemExit(
            f"estimated cost CNY {estimated_cost:.6f} exceeds cap "
            f"{args.max_estimated_cost_cny:.6f}"
        )
    if prior_cost + estimated_cost > args.max_cumulative_cost_cny:
        raise SystemExit(
            f"cumulative estimated upper bound CNY {prior_cost + estimated_cost:.6f} "
            f"exceeds cap {args.max_cumulative_cost_cny:.6f}"
        )
    if not args.live or not selected:
        return

    api_key = os.getenv(args.api_key_env)
    if not api_key:
        raise SystemExit(f"missing API key environment variable {args.api_key_env}")

    all_rows = list(existing_rows)
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    for manifest_row in selected:
        sample_id = manifest_row["sample_id"]
        image_path = args.image_dir / f"{sample_id}.jpg"
        usage_payload: dict[str, Any] | None = None
        try:
            response = call_model(
                api_key,
                args.base_url,
                args.model,
                args.protocol,
                manifest_row,
                image_path,
                args.max_output_tokens,
                args.timeout,
                args.gemini_thinking_budget,
            )
            content = response["content"]
            usage_payload = {
                "sample_id": sample_id,
                "model": args.model,
                "protocol": args.protocol,
                "request_id": response.get("request_id"),
                "status": "ok",
                "prompt_tokens": response.get("prompt_tokens"),
                "completion_tokens": response.get("completion_tokens"),
                "total_tokens": response.get("total_tokens"),
            }
            raw_payload = {**usage_payload, "content": content}
            (args.raw_dir / f"{sample_id}.json").write_text(
                json.dumps(raw_payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            probes = parse_probe_payload(content)
            validate_probe_content(probes, manifest_row)
            all_rows.extend(build_probe_rows(manifest_row, probes, args.model))
            write_probe_rows(args.output, all_rows)
            append_usage(args.usage_log, usage_payload)
            print(f"{sample_id}: wrote 3 draft probes")
        except Exception as error:
            error_payload = {
                "sample_id": sample_id,
                "model": args.model,
                "status": "content_rejected" if usage_payload else "api_error",
                "error_type": type(error).__name__,
                "error": str(error)[:500],
            }
            if usage_payload:
                error_payload = {**usage_payload, **error_payload}
            append_usage(args.usage_log, error_payload)
            if usage_payload and args.continue_on_rejection:
                print(f"{sample_id}: content rejected: {error}")
                continue
            raise


if __name__ == "__main__":
    main()
