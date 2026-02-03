#!/usr/bin/env python3
"""
Phase-2 cadence & beauty polish.

IMPORTANT DESIGN:
- Phase 2 is for cadence/beauty ONLY and must not "decide" divine pronouns.
- Divine pronoun capitalization is handled downstream by KJV normalization
  (your Option 2: caps + decaps by KJV), not here.

This script therefore avoids pronoun heuristics, with ONLY a few narrow,
explicitly requested, deterministic corrections:
  - "The Lord, the God" -> "The LORD, the God"
  - If "Pharaoh" appears: "His servants" -> "his servants"
  - "by my name" -> "by My name"

Input JSONL:
{"ref":"EXOD 1:1","translation":"..."}

Targets JSONL:
{"ref":"EXOD 1:1"}

Output JSONL matches input.
"""

import argparse
import json
import os
import re
import time
from typing import Dict, List, Set, Tuple, Optional

from openai import OpenAI


# ----------------------------
# IO helpers
# ----------------------------

def load_targets_jsonl(path: str) -> Set[str]:
    targets: Set[str] = set()
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            ref = (obj.get("ref") or "").strip()
            if not ref:
                raise ValueError(f"{path}:{i} missing 'ref'")
            targets.add(ref)
    return targets


def load_book_jsonl(path: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "ref" not in obj or "translation" not in obj:
                raise ValueError(f"{path}:{i} missing keys 'ref' and/or 'translation'")
            rows.append({"ref": str(obj["ref"]).strip(), "translation": str(obj["translation"])})
    return rows


def read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


# ----------------------------
# Guards
# ----------------------------

def normalize_space(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def similarity_guard(original: str, revised: str) -> Tuple[bool, str]:
    """
    Guard against model drift into commentary/headings/verse numbers or huge length changes.
    """
    o = normalize_space(original)
    r = normalize_space(revised)

    if not r:
        return False, "empty_output"

    lower = r.lower()
    if r.startswith("#") or lower.startswith(("note:", "commentary:", "explanation:", "translator", "translation note")):
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


# ----------------------------
# Deterministic enforcement (NO general pronoun changes)
# ----------------------------

def enforce_no_internal_line_breaks(text: str) -> str:
    # Replace embedded newlines with spaces and normalize whitespace
    t = re.sub(r"[\r\n]+", " ", text)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def enforce_between_from(text: str) -> str:
    t = text
    t = re.sub(r"\bseparated between\b", "separated", t, flags=re.IGNORECASE)
    t = re.sub(r"\bdivid(e|ing) between\b", r"divid\1", t, flags=re.IGNORECASE)

    t = re.sub(
        r"\bto divide\s+between\s+([^,;]+?)\s+and\s+between\s+([^,;]+?)\b",
        r"to divide \1 from \2",
        t,
        flags=re.IGNORECASE,
    )
    t = re.sub(
        r"\bto divide\s+([^,;]+?)\s+and\s+between\s+([^,;]+?)\b",
        r"to divide \1 from \2",
        t,
        flags=re.IGNORECASE,
    )
    t = re.sub(
        r"\bseparated\s+between\s+([^,;]+?)\s+and\s+the\s+([^,;]+?)\b",
        r"separated \1 from the \2",
        t,
        flags=re.IGNORECASE,
    )
    t = re.sub(
        r"\bseparated\s+between\s+([^,;]+?)\s+and\s+([^,;]+?)\b",
        r"separated \1 from \2",
        t,
        flags=re.IGNORECASE,
    )
    return t


def enforce_compound_numbers(text: str) -> str:
    t = text
    t = re.sub(
        r"\b(sixty|seventy|eighty|ninety)\s+and\s+(one|two|three|four|five|six|seven|eight|nine)\b",
        r"\1-\2",
        t,
        flags=re.IGNORECASE,
    )
    t = re.sub(
        r"\b(twenty|thirty|forty|fifty)\s+and\s+(one|two|three|four|five|six|seven|eight|nine)\b",
        r"\1-\2",
        t,
        flags=re.IGNORECASE,
    )
    return t


def enforce_lord_caps(text: str) -> str:
    """
    Normalize YHWH renderings to LORD in common English phrases while protecting 'Lord GOD'.
    """
    t = text

    # Preserve "Lord GOD" phrases as-is
    t = re.sub(r"\bLord GOD\b", "Lord GOD", t)
    t = re.sub(r"\bthe Lord GOD\b", "the Lord GOD", t)

    # Common narrative/speech formulas
    patterns = [
        (r"\bAnd the Lord said\b", "And the LORD said"),
        (r"\bThen the Lord said\b", "Then the LORD said"),
        (r"\bNow the Lord said\b", "Now the LORD said"),
        (r"\bThus says the Lord\b", "Thus says the LORD"),
        (r"\bThus saith the Lord\b", "Thus saith the LORD"),
        (r"\bthe Lord said\b", "the LORD said"),
        (r"\bthe Lord spoke\b", "the LORD spoke"),
        (r"\bthe Lord called\b", "the LORD called"),
        (r"\bthe Lord commanded\b", "the LORD commanded"),
    ]
    for pat, rep in patterns:
        t = re.sub(pat, rep, t)

    # Punctuation variants
    t = re.sub(r"\bThe Lord,\b", "The LORD,", t)
    t = re.sub(r"\bthe Lord,\b", "the LORD,", t)

    # General "the Lord" -> "the LORD" (avoid "the LORD GOD")
    t = re.sub(r"\bthe Lord\b(?!\s+GOD\b)", "the LORD", t)

    # "Lord God" -> "LORD God"
    t = re.sub(r"\bLord God\b", "LORD God", t)

    # Keep "angel of the LORD" consistent
    t = re.sub(r"\bangel of the Lord\b", "angel of the LORD", t)

    return t


def enforce_yhwh_titlecase(text: str) -> str:
    """
    Fix cases like 'The Lord, the God of...' that should be 'The LORD, the God of...'
    Very narrow: only when 'the God' immediately follows.
    """
    t = text
    t = re.sub(r"\bThe Lord,\s+the God\b", "The LORD, the God", t)
    t = re.sub(r"\bthe Lord,\s+the God\b", "the LORD, the God", t)
    return t


def enforce_pharaoh_servants_pronoun(text: str) -> str:
    """
    If Pharaoh is explicit and we see 'His servants' in the same verse,
    force lowercase because antecedent is Pharaoh.
    Narrow + safe.
    """
    t = text
    if re.search(r"\bPharaoh\b", t) and re.search(r"\bHis servants\b", t):
        t = re.sub(r"\bHis servants\b", "his servants", t)
    return t


def enforce_by_my_name_my(text: str) -> str:
    """
    Fix 'by my name' -> 'by My name' (narrow).
    Only triggers for the specific phrase to avoid broad pronoun heuristics.
    """
    return re.sub(r"\bby my name\b", "by My name", text)


def validate_enforcement(text: str) -> None:
    if "angel of the LORD" in text and "angel of the Lord" in text:
        raise ValueError("Mixed 'angel of the LORD' and 'angel of the Lord' after enforcement")


def apply_enforcement(text: str) -> str:
    t = text
    t = enforce_no_internal_line_breaks(t)

    # LORD/YHWH conventions
    t = enforce_lord_caps(t)
    t = enforce_yhwh_titlecase(t)

    # Narrow, safe, explicitly requested fixes
    t = enforce_pharaoh_servants_pronoun(t)
    t = enforce_by_my_name_my(t)

    # Other style normalizations
    t = enforce_between_from(t)
    t = enforce_compound_numbers(t)

    validate_enforcement(t)
    return t


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase-2 cadence/beauty polish (no pronoun heuristics).")
    parser.add_argument("--in", dest="inp", required=True)
    parser.add_argument("--out", dest="out", required=True)
    parser.add_argument("--targets", required=True)
    parser.add_argument("--charter", required=True)
    parser.add_argument("--model", default="gpt-5.1")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max_output_tokens", type=int, default=300)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--enforce", action="store_true", help="Apply deterministic enforcement rules")
    parser.add_argument("--enforce_only", action="store_true", help="Skip model calls; enforcement-only pass")
    args = parser.parse_args()

    targets = load_targets_jsonl(args.targets)
    rows = load_book_jsonl(args.inp)
    charter = read_text_file(args.charter)

    system_prompt = (
        charter
        + "\n\nPHASE-2 OPERATIONAL RULES\n"
        + "- You are NOT translating Hebrew or Greek.\n"
        + "- Revise English ONLY for cadence, beauty, and recognizability.\n"
        + "- Meaning must remain unchanged.\n"
        + "- Do NOT add verse numbers, headings, or commentary.\n"
        + "- Output ONLY the revised verse text.\n"
    )

    client: Optional[OpenAI] = None
    if not args.enforce_only:
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is missing or empty (required unless --enforce_only)")
        client = OpenAI(api_key=api_key)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    changed_by_model = 0
    guard_blocked = 0
    enforcement_changed = 0
    model_calls = 0

    with open(args.out, "w", encoding="utf-8") as fout:
        for row in rows:
            ref = row["ref"]
            original = row["translation"]
            out_text = original

            do_model = (not args.enforce_only) and (ref in targets)

            if do_model:
                assert client is not None
                user_prompt = (
                    f"REFERENCE: {ref}\n"
                    f"Revise the following English verse for cadence and beauty.\n"
                    f"Do not change meaning or theology.\n"
                    f"Output only the revised verse text.\n\n"
                    f"ORIGINAL ENGLISH:\n{original}\n"
                )

                resp = client.responses.create(
                    model=args.model,
                    input=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=args.temperature,
                    max_output_tokens=args.max_output_tokens,
                )
                model_calls += 1
                candidate = (resp.output_text or "").strip()

                if args.enforce:
                    enforced = apply_enforcement(candidate)
                    if normalize_space(enforced) != normalize_space(candidate):
                        enforcement_changed += 1
                    candidate = enforced

                ok, reason = similarity_guard(original, candidate)
                if not ok:
                    print(f"WARNING: Guard blocked {ref}: {reason}")
                    out_text = original
                    guard_blocked += 1
                else:
                    out_text = candidate
                    if normalize_space(out_text) != normalize_space(original):
                        changed_by_model += 1

            else:
                # Non-target or enforce-only
                if args.enforce:
                    enforced = apply_enforcement(out_text)
                    if normalize_space(enforced) != normalize_space(out_text):
                        enforcement_changed += 1
                    out_text = enforced

            fout.write(json.dumps({"ref": ref, "translation": out_text}, ensure_ascii=False) + "\n")

            if args.sleep and do_model:
                time.sleep(args.sleep)

    print(
        "Phase-2 complete. "
        f"Changed(by model): {changed_by_model} | "
        f"Guard-blocked: {guard_blocked} | "
        f"Enforcement-changed: {enforcement_changed} | "
        f"Model-calls: {model_calls}"
    )


if __name__ == "__main__":
    main()
