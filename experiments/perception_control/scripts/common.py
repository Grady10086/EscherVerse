from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from statistics import NormalDist
from typing import Any, Iterable


CATEGORY_NAMES = {
    1: "Object Permanence and Occlusion Tracking",
    2: "Dynamic Spatial Relationships",
    3: "Action and Intent-Driven Spatial Reasoning",
    4: "Predictive and Counterfactual Spatial Reasoning",
    5: "Object Deformation and State Transition",
    6: "Egocentric vs. Allocentric Reference Frames",
}

PROBE_TYPES = ("entity", "action_event", "simple_relation")
PROBE_SUBTYPES_BY_TYPE = {
    "entity": ("object_actor_presence",),
    "action_event": ("action_recognition", "temporal_order"),
    "simple_relation": ("orientation", "simple_spatial_relation"),
}
PROBE_SUBTYPES = tuple(
    subtype
    for probe_type in PROBE_TYPES
    for subtype in PROBE_SUBTYPES_BY_TYPE[probe_type]
)
OPTION_LETTERS = ("A", "B", "C", "D")


def canonical_category(raw: str) -> tuple[int, str]:
    match = re.match(r"\s*Category\s+([1-6])\s*:", raw or "", re.IGNORECASE)
    if not match:
        raise ValueError(f"Cannot canonicalize category: {raw!r}")
    category_id = int(match.group(1))
    return category_id, CATEGORY_NAMES[category_id]


def canonical_scene_type(raw: str) -> str:
    value = re.sub(r"[^a-z]", "", (raw or "").lower())
    if value in {"humancentric", "humancentered", "humanoriented", "peopleoriented"}:
        return "Human-Centric"
    if value in {"objectcentric", "objectcentered"}:
        return "Object-Centric"
    return (raw or "Unknown").strip() or "Unknown"


def canonical_question_type(raw: str) -> str:
    value = re.sub(r"[^a-z]", "", (raw or "").lower())
    mapping = {
        "singlechoice": "Single-Choice",
        "multipleselect": "Multiple-Select",
        "truefalse": "True/False",
        "fillintheblank": "Fill-in-the-Blank",
    }
    return mapping.get(value, (raw or "Unknown").strip() or "Unknown")


def stable_digest(parts: Iterable[Any]) -> str:
    payload = "\x1f".join(str(part) for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def wilson_interval(successes: int, total: int, confidence: float = 0.95) -> tuple[float, float]:
    if total <= 0:
        return float("nan"), float("nan")
    alpha = 1.0 - confidence
    z = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    margin = (
        z
        * math.sqrt((p * (1.0 - p) + z * z / (4.0 * total)) / total)
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def exact_mcnemar_p_value(b: int, c: int) -> float:
    discordant = b + c
    if discordant == 0:
        return 1.0
    lower_tail = sum(math.comb(discordant, k) for k in range(0, min(b, c) + 1))
    return min(1.0, 2.0 * lower_tail / (2**discordant))


def count_values(rows: Iterable[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get(key, "Unknown")) for row in rows).items()))
