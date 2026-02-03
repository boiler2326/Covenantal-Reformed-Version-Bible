#!/usr/bin/env python3
"""
KJV-gated pronoun normalization (FULL caps + decaps) - NO API.

Input book JSONL:
{"ref":"EXOD 4:27","translation":"..."}

Input KJV JSONL:
{"ref":"EXOD 4:27","kjv":"..."}

Output:
- normalized book JSONL
- review targets JSONL (mixed verses, missing KJV, etc.)
- optional stats JSON

Rules (Option 2):
- divine_only (KJV has He/Him/His/Himself and no lowercase): force-cap in your verse
- human_only (KJV has he/him/his/himself and no divine caps): force-decap in your verse
- mixed: default skip + review (recommended)
- none: do nothing

Tokens affected:
he/him/his/himself <-> He/Him/His/Himself
"""

import argparse
import json
import os
import re
from typing import Dict, List

DIV_CAP_RE = re.compile(r"\b(He|Him|His|Himself)\b")
HUM_LOW_RE = re.compile(r"\b(he|him|his|himself)\b")

LOW_TO_CAP = {"he": "He", "him": "Him", "his": "His", "himself": "Himself"}
CAP_TO_LOW = {v: k for k, v in LOW_TO_CAP.items()}

LOW_REPLACE_RE = re.compile(r"\b(he|him|his|himself)\b")
CAP_REPLACE_RE = re.compile(r"\b(He|Him|His|Himself)\b")


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


def force_cap(text: str) -> str:
    def repl(m: re.Match) -> str:
        return LOW_TO_CAP[m.group(0)]
    return LOW_REPLACE_RE.sub(repl, text)


def force_decap(text: str) -> str:
    def repl(m: re.Match) -> str:
        return CAP_TO_LOW[m.group(0)]
    return CAP_REPLACE_RE.sub(repl, text)


def main() -> None:
    ap = argparse.ArgumentParser(description="KJV pronoun normalization (caps + decaps), no API.")
    ap.add_argument("--book_in", required=True)
    ap.add_argument("--kjv", required=True)
    ap.add_argument("--book_out", required=True)
    ap.add_argument("--review_out", required=True)
    ap.add_argument("--stats_out", required=False)
    ap.add_argument(
        "--mixed_policy",
        choices=["skip", "cap_only", "normalize_anyway"],
        default="skip",
    )
    args = ap.parse_args()

    kjv_map = load_jsonl_map(args.kjv, "ref", "kjv")

    os.makedirs(os.path.dirname(args.book_out) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.review_out) or ".", exist_ok=True)

    review: List[dict] = []

    total = 0
    changed = 0
    missing_kjv = 0

    divine_only = 0
    human_only = 0
    mixed = 0
    none = 0
    mixed_changed = 0

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
                review.append({"ref": ref, "reason": "missing_kjv_ref"})
                fout.write(json.dumps({"ref": ref, "translation": text}, ensure_ascii=False) + "\n")
                continue

            cls = classify_kjv(kjv_text)
            new_text = text

            if cls == "divine_only":
                divine_only += 1
                new_text = force_cap(text)

            elif cls == "human_only":
                human_only += 1
                new_text = force_decap(text)

            elif cls == "mixed":
                mixed += 1
                if args.mixed_policy == "skip":
                    review.append({"ref": ref, "reason": "mixed_pronouns_in_kjv"})
                    new_text = text
                elif args.mixed_policy == "cap_only":
                    new_text = force_cap(text)
                    if new_text != text:
                        mixed_changed += 1
                    review.append({"ref": ref, "reason": "mixed_pronouns_in_kjv_cap_only"})
                else:
                    tmp = force_cap(text)
                    new_text = force_decap(tmp)
                    if new_text != text:
                        mixed_changed += 1
                    review.append({"ref": ref, "reason": "mixed_pronouns_in_kjv_normalize_anyway"})

            else:
                none += 1
                new_text = text

            if new_text != text:
                changed += 1

            fout.write(json.dumps({"ref": ref, "translation": new_text}, ensure_ascii=False) + "\n")

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
        "mixed_policy": args.mixed_policy,
        "mixed_changed": mixed_changed,
        "review_targets": len(review),
    }

    if args.stats_out:
        os.makedirs(os.path.dirname(args.stats_out) or ".", exist_ok=True)
        with open(args.stats_out, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

    with open(args.review_out, "w", encoding="utf-8") as f:
        for item in review:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("KJV pronoun normalization complete:")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
