#!/usr/bin/env python3

import argparse
import json
import os
import re
import time
from typing import Dict, List, Set, Tuple

from openai import OpenAI


def load_targets_jsonl(path: str) -> Set[str]:
    targets: Set[str] = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            ref = obj.get("ref", "").strip()
            if not ref:
                raise ValueError(f"targets.jsonl line missing 'ref': {line[:120]}")
            targets.add(ref)
    return targets


def load_phase1_jsonl(path: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "ref" not in obj or "translation" not in obj:
                raise ValueError(f"Phase-1 jsonl missing keys at line {i}: {line[:200]}")
            rows.append({"ref": obj["ref"], "translation": obj["translation"]})
    return rows


def read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def normalize_space(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def similarity_guard(original: str, revised: str) -> Tuple[bool, str]:
    """
    Lightweight guard against meaning drift:
    - Ensure revised is non-empty and not wildly longer/shorter.
    - Ensure revised doesn't introduce bracketed commentary or headings.
    This is NOT a semantic proof; it just blocks obvious failures.
    """
    o = normalize_space(original)
    r = normalize_space(revised)

    if not r:
        return False, "empty_output"

    # Block headings / commentary-y artifacts
    if r.startswith("#") or r.lower().startswith(("note:", "commentary:", "explanation:", "translator")):
        return False, "commentary_or_heading"

    # Block verse numbers being added at the start
    if re.match(r"^\d+\s", r):
        return False, "added_verse_number"

    # Length ratio guard (very permissive, just prevents extreme drift)
    if len(o) >= 20:
        ratio = len(r) / max(1, len(o))
        if ratio < 0.60:
            return False, f"too_short_ratio_{ratio:.2f}"
        if ratio > 1.60:
            return False, f"too_long_ratio_{ratio:.2f}"

    return True, "ok"


def main():
    parser = argparse.ArgumentParser(
        description="Phase-2 polish pass: improves cadence/beauty without changing meaning, only for targeted refs."
    )
    parser.add_argument("--in", dest="inp", required=True, help="Phase-1 output JSONL (ref, translation)")
    parser.add_argument("--out", dest="out", required=True, help="Phase-2 output JSONL (ref, translation)")
    parser.add_argument("--targets", required=True, help="JSONL list of refs to polish (one {'ref':...} per line)")
    parser.add_argument("--charter", required=True, help="Phase-2 charter text file")
    parser.add_argument("--model", default="gpt-5.1", help="Model to use for polishing")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--sleep", type=float, default=0.0, help="Optional delay between requests (seconds)")
    parser.add_argument("--max_output_tokens", type=int, default=300)
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing or empty. Check GitHub repo secret OPENAI_API_KEY.")

    client = OpenAI(api_key=api_key)

    targets = load_targets_jsonl(args.targets)
    phase1_rows = load_phase1_jsonl(args.inp)
    phase2_charter = read_text_file(args.charter)

    # Phase-2 system prompt: charter + strict constraints
    system_prompt = (
        phase2_charter
        + "\n\n"
        + "PHASE-2 OPERATIONAL RULES\n"
        + "- You are NOT translating Hebrew/Greek.\n"
        + "- You are revising an existing English verse ONLY for cadence, beauty, and recognizability.\n"
        + "- Meaning must remain unchanged.\n"
        + "- Do NOT add verse numbers, headings, notes, or commentary.\n"
        + "- Output ONLY the revised verse text.\n"
    )

    # Make sure output directory exists
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    changed = 0
    warned = 0

    with open(args.out, "w", encoding="utf-8") as fout:
        for row in phase1_rows:
            ref = row["ref"]
            original = row["translation"]

            if ref not in targets:
                fout.write(json.dumps({"ref": ref, "translation": original}, ensure_ascii=False) + "\n")
                continue

            user_prompt = (
                f"REFERENCE: {ref}\n"
                f"Revise the following English verse for cadence, beauty, and recognizability.\n"
                f"Do not change meaning. Do not add or remove theological content.\n"
                f"Output only the revised verse text.\n\n"
                f"ORIGINAL ENGLISH:\n{original}\n"
            )

            response = client.responses.create(
                model=args.model,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=args.temperature,
                max_output_tokens=args.max_output_tokens,
            )

            revised = response.output_text.strip()

            ok, reason = similarity_guard(original, revised)
            if not ok:
                # If guard fails, keep original and log a warning to Action logs.
                print(f"WARNING: Phase-2 guard blocked {ref}: {reason}. Keeping Phase-1 text.")
                warned += 1
                revised = original
            else:
                if normalize_space(revised) != normalize_space(original):
                    changed += 1

            fout.write(json.dumps({"ref": ref, "translation": revised}, ensure_ascii=False) + "\n")

            if args.sleep:
                time.sleep(args.sleep)

    print(f"Phase-2 complete. Targets: {len(targets)} | Changed: {changed} | Guard-blocked: {warned}")


if __name__ == "__main__":
    main()
```0
