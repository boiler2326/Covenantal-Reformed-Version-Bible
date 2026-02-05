#!/usr/bin/env python3
import json
import csv
from typing import Dict, List, Tuple

def read_approved(review_csv: str) -> Dict[str, str]:
    """
    Returns mapping: ref -> approved_translation_text
    Only rows with decision == APPROVE are applied.
    If multiple APPROVE rows exist for same ref, last one wins.
    """
    approved: Dict[str, str] = {}
    with open(review_csv, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        required = {"decision", "ref", "original", "suggested"}
        if not required.issubset(set(r.fieldnames or [])):
            raise ValueError(f"CSV must contain columns: {sorted(required)}")

        for row in r:
            decision = (row.get("decision") or "").strip().upper()
            if decision != "APPROVE":
                continue
            ref = (row.get("ref") or "").strip()
            suggested = row.get("suggested") or ""
            if not ref:
                continue
            approved[ref] = suggested
    return approved

def apply(in_jsonl: str, out_jsonl: str, approved: Dict[str, str]) -> Tuple[int, int]:
    total = 0
    changed = 0
    with open(in_jsonl, "r", encoding="utf-8") as fin, open(out_jsonl, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            obj = json.loads(line)
            total += 1
            ref = obj.get("ref", "")
            if ref in approved:
                before = obj.get("translation", "")
                after = approved[ref]
                if isinstance(before, str) and before != after:
                    obj["translation"] = after
                    changed += 1
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    return total, changed

def main(in_jsonl: str, review_csv: str, out_jsonl: str):
    approved = read_approved(review_csv)
    total, changed = apply(in_jsonl, out_jsonl, approved)
    print(f"Total lines: {total}")
    print(f"Approved edits applied: {len(approved)}")
    print(f"Lines changed: {changed}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 4:
        print("Usage: python scripts/apply_caps.py input.jsonl review.csv output.jsonl")
        raise SystemExit(2)
    main(sys.argv[1], sys.argv[2], sys.argv[3])
