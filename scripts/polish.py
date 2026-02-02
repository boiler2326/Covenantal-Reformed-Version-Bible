#!/usr/bin/env python3

import argparse
import json
import os
import re
import time
from typing import Dict, List, Set, Tuple

from openai import OpenAI


# -------------------------
# IO helpers
# -------------------------

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
                raise ValueError("targets.jsonl line missing 'ref'")
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
                raise ValueError(f"Phase-1 jsonl missing keys at line {i}")
            rows.append({"ref": obj["ref"], "translation": obj["translation"]})
    return rows


def read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def normalize_space(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


# -------------------------
# Safety / drift guard
# -------------------------

def similarity_guard(original: str, revised: str) -> Tuple[bool, str]:
    o = normalize_space(original)
    r = normalize_space(revised)

    if not r:
        return False, "empty_output"

    if r.startswith("#") or r.lower().startswith(("note:", "commentary:", "explanation:", "translator")):
        return False, "commentary_or_heading"

    if re.match(r"^\d+\s", r):
        return False, "added_verse_number"

    if len(o) >= 20:
        ratio = len(r) / max(1, len(o))
        if ratio < 0.60:
            return False, "too_short"
        if ratio > 1.60:
            return False, "too_long"

    return True, "ok"


# -------------------------
# Enforcement rules
# -------------------------

def validate_enforcement(text: str) -> None:
    # Hard fail: mixed LORD/Lord in the same phrase context
    if "angel of the LORD" in text and "angel of the Lord" in text:
        raise ValueError("Mixed 'angel of the LORD' and 'angel of the Lord' after enforcement")


def enforce_between_from(text: str) -> str:
    # Normalize common Hebrew calque: "separated between X and Y" -> "separated X from Y"
    # First, remove "between" after "separated/divide" when it appears as a direct calque.
    text = re.sub(r"\bseparated between\b", "separated", text, flags=re.IGNORECASE)
    text = re.sub(r"\bdivid(e|ing) between\b", r"divid\1", text, flags=re.IGNORECASE)

    # Specific patterns seen in Genesis:
    text = re.sub(
        r"\bto divide\s+between\s+([^,;]+?)\s+and\s+between\s+([^,;]+?)\b",
        r"to divide \1 from \2",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\bto divide\s+([^,;]+?)\s+and\s+between\s+([^,;]+?)\b",
        r"to divide \1 from \2",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\bseparated\s+between\s+([^,;]+?)\s+and\s+the\s+([^,;]+?)\b",
        r"separated \1 from the \2",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\bseparated\s+between\s+([^,;]+?)\s+and\s+([^,;]+?)\b",
        r"separated \1 from \2",
        text,
        flags=re.IGNORECASE,
    )
    return text


def enforce_lord_caps(text: str) -> str:
    # Keep "Lord GOD" as-is (Adonai YHWH style) if it appears.
    # Normalize "Lord God" -> "LORD God" (YHWH Elohim pattern)
    text = re.sub(r"\bLord God\b", "LORD God", text)

    # Normalize narrative formula "And the Lord said" -> "And the LORD said"
    text = re.sub(r"\bAnd the Lord said\b", "And the LORD said", text)

    # General: "the Lord" -> "the LORD" (but avoid changing "Lord GOD")
    text = re.sub(r"\bthe Lord\b(?!\s+GOD\b)", "the LORD", text)

    # Keep "angel of the LORD" consistent
    text = re.sub(r"\bangel of the Lord\b", "angel of the LORD", text)

    return text


def enforce_compound_numbers(text: str) -> str:
    # Fix explicit "... sixty and five ..." patterns into "... sixty-five ..."
    text = re.sub(
        r"\b(sixty|seventy|eighty|ninety)\s+and\s+(one|two|three|four|five|six|seven|eight|nine)\b",
        r"\1-\2",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\b(twenty|thirty|forty|fifty)\s+and\s+(one|two|three|four|five|six|seven|eight|nine)\b",
        r"\1-\2",
        text,
        flags=re.IGNORECASE,
    )
    return text


def enforce_reverential_pronouns(text: str) -> str:
    """
    Conservative reverential caps:
    - Only capitalize He/Him/His/Himself when 'God' or 'LORD' appears in the SAME verse,
      making the antecedent explicit.
    """
    if not re.search(r"\b(God|LORD|Lord GOD|the LORD)\b", text):
        return text

    # Only target clear third-person masculine pronouns.
    text = re.sub(r"\bhe\b", "He", text)
    text = re.sub(r"\bhim\b", "Him", text)
    text = re.sub(r"\bhis\b", "His", text)
    text = re.sub(r"\bhimself\b", "Himself", text)
    return text


def apply_enforcement(text: str) -> str:
    text = enforce_lord_caps(text)
    text = enforce_between_from(text)
    text = enforce_compound_numbers(text)
    text = enforce_reverential_pronouns(text)
    validate_enforcement(text)
    return text


# -------------------------
# Main
# -------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase-2 polish pass")
    parser.add_argument("--in", dest="inp", required=True, help="Phase-1 input JSONL (ref, translation)")
    parser.add_argument("--out", dest="out", required=True, help="Phase-2 output JSONL (ref, translation)")
    parser.add_argument("--targets", required=True, help="JSONL targets list (one {'ref':...} per line)")
    parser.add_argument("--charter", required=True, help="Phase-2 charter text file")
    parser.add_argument("--model", default="gpt-5.1")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--max_output_tokens", type=int, default=300)
    parser.add_argument("--enforce", action="store_true", help="Apply deterministic enforcement rules to all verses")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing or empty")

    client = OpenAI(api_key=api_key)

    targets = load_targets_jsonl(args.targets)
    phase1_rows = load_phase1_jsonl(args.inp)
    phase2_charter = read_text_file(args.charter)

    system_prompt = (
        phase2_charter
        + "\n\n"
        + "PHASE-2 OPERATIONAL RULES\n"
        + "- You are NOT translating Hebrew or Greek.\n"
        + "- Revise English ONLY for cadence, beauty, and recognizability.\n"
        + "- Meaning must remain unchanged.\n"
        + "- Do NOT add verse numbers, headings, or commentary.\n"
        + "- Output ONLY the revised verse text.\n"
    )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    changed = 0
    blocked = 0

    with open(args.out, "w", encoding="utf-8") as fout:
        for row in phase1_rows:
            ref = row["ref"]
            original = row["translation"]

            # Default output is the Phase-1 text
            out_text = original

            # Only call the model for verses in the targets list
            if ref in targets:
                user_prompt = (
                    f"REFERENCE: {ref}\n"
                    f"Revise the following English verse for cadence and beauty.\n"
                    f"Do not change meaning or theology.\n"
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

                if ok:
                    out_text = revised
                    if normalize_space(out_text) != normalize_space(original):
                        changed += 1
                else:
                    print(f"WARNING: Guard blocked {ref}: {reason}")
                    blocked += 1
                    out_text = original

            # Apply deterministic enforcement to ALL verses (targets and non-targets) if enabled
            if args.enforce:
                out_text = apply_enforcement(out_text)

            fout.write(json.dumps({"ref": ref, "translation": out_text}, ensure_ascii=False) + "\n")

            if args.sleep:
                time.sleep(args.sleep)

    print(f"Phase-2 complete. Targets: {len(targets)} | Changed: {changed} | Guard-blocked: {blocked}")


if __name__ == "__main__":
    main()
