#!/usr/bin/env python3
"""
KJV-gated pronoun capitalization pass (NO API).

Input:
- Your book JSONL: {"ref":"EXOD 4:27","translation":"..."}
- KJV JSONL: {"ref":"EXOD 4:27","kjv":"..."}

Output:
- Fixed JSONL (same schema as input)
- Review targets JSONL listing refs that are ambiguous ("mixed" in KJV)
- Stats printed to stdout

Gating logic:
- If KJV verse contains ONLY divine-caps pronouns (He/Him/His/Himself) and NO lowercase he/him/his/himself:
  -> capitalize those pronouns in your verse.
- If KJV verse contains ONLY lowercase he/him/his/himself and NO divine-caps:
  -> lowercase those pronouns in your verse.
- If KJV verse contains BOTH (mixed):
  -> do NOT change; add to review list.
- If KJV verse contains neither:
  -> do NOT change.

This intentionally avoids the “endless rerun” trap: mixed verses require human judgment.
"""

import argparse
import json
import os
import re
from typing import Dict, List, Tuple

DIV_CAP_RE = re.compile(r"\b(He|Him|His|Himself)\b")
HUM_LOW_RE = re.compile(r"\b(he|him|his|himself)\b")

LOW_TO_CAP = {"he": "He", "him": "Him", "his": "His", "himself": "Himself"}
CAP_TO_LOW = {v: k for k, v in LOW_TO_CAP.items()}


def load_jsonl_map(path: str, key_field: str, val_field: str) -> Dict[str, str]:
    m: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            key = (obj.get(key_field) or "").strip()
            if not key:
                raise ValueError(f"{path}:{line_no} missing '{key_field}'")
            val = obj.get(val_field)
            if val is None:
                raise ValueError(f"{path}:{line_no} missing '{val_field}'")
            m[key] = str(val)
    return m


def classify_kjv(text: str) -> str:
    has_div = bool(DIV_CAP_RE.search(text))
    has_hum = bool(HUM_LOW_RE.search(text))
    if has_div and has_hum:
        return "mixed"
    if has_div:
        return "divine_only"
    if has_hum:
        return "human_only"
    return "none"


def cap_all(text: str) -> str:
    def repl(m: re.Match) -> str:
        return LOW_TO_CAP[m.group(0)]
    return re.sub(r"\b(he|him|his|himself)\b", repl, text)


def decap_all(text: str) -> str:
    def repl(m: re.Match) -> str:
        return CAP_TO_LOW[m.group(0)]
    return re.sub(r"\b(He|Him|His|Himself)\b", repl, text)


def main() -> None:
    ap = argparse.ArgumentParser(description="KJV-gated pronoun pass (no API).")
    ap.add_argument("--book_in", required=True, help="Input book JSONL (e.g., output_phase2/exodus.jsonl)")
    ap.add_argument("--kjv", required=True, help="KJV JSONL (e.g., sources/kjv/kjv.jsonl)")
    ap.add_argument("--book_out", required=True, help="Output JSONL (fixed)")
    ap.add_argument("--review_out", required=True, help="Review targets JSONL (mixed verses)")
    ap.add_argument("--stats_out", required=False, help="Optional stats JSON file")
    args = ap.parse_args()

    kjv_map = load_jsonl_map(args.kjv, "ref", "kjv")

    os.makedirs(os.path.dirname(args.book_out) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.review_out) or ".", exist_ok=True)

    review: List[dict] = []

    total = 0
    changed = 0
    mixed = 0
    none = 0
    missing_kjv = 0
    divine_only = 0
    human_only = 0

    with open(args.book_in, "r", encoding="utf-8") as fin, open(args.book_out, "w", encoding="utf-8") as fout:
        for line_no, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue
            total += 1
            obj = json.loads(line)
            ref = (obj.get("ref") or "").strip()
            text = obj.get("translation", "")
            if not ref:
                raise ValueError(f"{args.book_in}:{line_no} missing 'ref'")

            kjv_text = kjv_map.get(ref)
            if not kjv_text:
                missing_kjv += 1
                fout.write(json.dumps({"ref": ref, "translation": text}, ensure_ascii=False) + "\n")
                continue

            cls = classify_kjv(kjv_text)
            new_text = text

            if cls == "divine_only":
                divine_only += 1
                new_text = cap_all(text)
            elif cls == "human_only":
                human_only += 1
                new_text = decap_all(text)
            elif cls == "mixed":
                mixed += 1
                review.append({"ref": ref, "reason": "mixed_pronouns_in_kjv"})
                new_text = text
            else:
                none += 1
                new_text = text

            if new_text != text:
                changed += 1

            fout.write(json.dumps({"ref": ref, "translation": new_text}, ensure_ascii=False) + "\n")

    with open(args.review_out, "w", encoding="utf-8") as f:
        for t in review:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    stats = {
        "total": total,
        "changed": changed,
        "missing_kjv": missing_kjv,
        "classified": {
            "divine_only": divine_only,
            "human_only": human_only,
            "mixed": mixed,
            "none": none,
        },
        "review_targets": len(review),
    }

    if args.stats_out:
        os.makedirs(os.path.dirname(args.stats_out) or ".", exist_ok=True)
        with open(args.stats_out, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

    print("Pronoun pass complete:")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
