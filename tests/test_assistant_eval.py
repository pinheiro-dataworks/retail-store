"""Keep the assistant eval set in sync with the exported context.

If the pipeline output changes (new numbers after retraining), the expected
answers in eval/assistant_questions.json must change too; this test fails
until they do.
"""

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ITEMS = json.loads((ROOT / "eval" / "assistant_questions.json").read_text(encoding="utf-8"))
ITEMS = ITEMS["items"]
CONTEXT = json.loads((ROOT / "web" / "data" / "assistant_context.json").read_text(encoding="utf-8"))


def _lookup(path: str):
    node = CONTEXT
    for part in path.split("."):
        node = node[int(part)] if isinstance(node, list) else node[part]
    return node


def _expected_text(item) -> str:
    """The text a correct answer would contain for this item."""
    if item["id"] == "top_region":
        return max(CONTEXT["regions"], key=lambda r: float(CONTEXT["regions"][r].strip("$B")))
    if item["id"] == "biggest_holiday":
        lifts = {k: float(v.strip("+%")) for k, v in CONTEXT["holiday_lift_by_event"].items()}
        return max(lifts, key=lifts.get)
    value = _lookup(item["source"])
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value}%"
    return str(value)


@pytest.mark.parametrize("item", [i for i in ITEMS if i["kind"] == "fact"], ids=lambda i: i["id"])
def test_expected_answer_matches_current_context(item):
    text = _expected_text(item)
    assert any(re.search(p, text, re.IGNORECASE) for p in item["accept"]), (text, item["accept"])


def test_eval_set_shape():
    ids = [i["id"] for i in ITEMS]
    assert len(ids) == len(set(ids))
    assert sum(i["kind"] == "refusal" for i in ITEMS) >= 2
    for i in ITEMS:
        assert i["accept"]
        for p in i["accept"]:
            re.compile(p)


def _load_scorer():
    import importlib.util

    path = ROOT / "scripts" / "eval_assistant.py"
    spec = importlib.util.spec_from_file_location("eval_assistant", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Real gpt-oss-120b replies (shortened) that the scorer used to reject.
TYPOGRAPHIC_REPLIES = [
    ("biggest_holiday", "The biggest lift comes from **Black Friday** (+103.4%)."),
    ("protocol", "Testing period: **2024‑07‑01 through 2024‑12‑21**."),
    ("refuse_store_week", "The dataset does not include store‑department‑date sales, "
                          "so I cannot give the exact figure."),
    ("refuse_far_future", "No figures are available for December 2025, so I cannot give "
                          "an estimate."),
]


@pytest.mark.parametrize(("item_id", "reply"), TYPOGRAPHIC_REPLIES)
def test_scorer_accepts_typographic_replies(item_id, reply):
    item = next(i for i in ITEMS if i["id"] == item_id)
    assert _load_scorer().passes(item, reply)
