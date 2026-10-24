"""Score the dashboard assistant against eval/assistant_questions.json.

Sends each question to a running /api/chat endpoint (local `vercel dev` or the
deployment) and checks the reply with the item's regular expressions:
fact items must contain the expected value, refusal items must decline.

    python scripts/eval_assistant.py --url https://retail-store-eta.vercel.app/api/chat
    python scripts/eval_assistant.py --url http://localhost:3000/api/chat --api-key gsk_...

The report is written to artifacts/assistant_eval.json.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QUESTIONS = ROOT / "eval" / "assistant_questions.json"
REPORT = ROOT / "artifacts" / "assistant_eval.json"


def ask(url: str, question: str, api_key: str | None, timeout: float = 60) -> str:
    payload = {"message": question} | ({"apiKey": api_key} if api_key else {})
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())["reply"]


def passes(item: dict, reply: str) -> bool:
    return any(re.search(p, reply, flags=re.IGNORECASE) for p in item["accept"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", required=True, help="full URL of the /api/chat endpoint")
    parser.add_argument("--api-key", help="Groq key, only if the deployment has no server key")
    parser.add_argument("--delay", type=float, default=6.5,
                        help="seconds between requests (stays under the default rate limit)")
    args = parser.parse_args()

    items = json.loads(QUESTIONS.read_text(encoding="utf-8"))["items"]
    results = []
    for i, item in enumerate(items):
        if i:
            time.sleep(args.delay)
        try:
            reply = ask(args.url, item["question"], args.api_key)
            ok = passes(item, reply)
        except (urllib.error.URLError, KeyError, TimeoutError) as exc:
            reply, ok = f"ERROR: {exc}", False
        results.append({"id": item["id"], "kind": item["kind"], "pass": ok, "reply": reply})
        print(f"[{'PASS' if ok else 'FAIL'}] {item['id']}")

    by_kind = {}
    for kind in ("fact", "refusal"):
        subset = [r for r in results if r["kind"] == kind]
        by_kind[kind] = sum(r["pass"] for r in subset) / len(subset)
    accuracy = sum(r["pass"] for r in results) / len(results)
    print(f"\naccuracy {accuracy:.0%}  (facts {by_kind['fact']:.0%}, "
          f"refusals {by_kind['refusal']:.0%})")

    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text(json.dumps({"url": args.url, "accuracy": accuracy, "by_kind": by_kind,
                                  "results": results}, indent=2), encoding="utf-8")
    return 0 if accuracy == 1 else 1


if __name__ == "__main__":
    sys.exit(main())
