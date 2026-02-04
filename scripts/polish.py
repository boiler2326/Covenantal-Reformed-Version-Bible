#!/usr/bin/env python3

import argparse
import json
import os
import re
import time
from typing import Dict, List, Set, Tuple, Optional

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
                raise ValueError("targets.jsonl line missing 'ref'")
            targets.add(ref)
    return targets


def load_phase_jsonl(path: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "ref" not in obj or "translation" not in obj:
                raise ValueError(f"Input jsonl missing keys at line {i}")
            rows.append({"ref": obj["ref"], "translation": obj["translation"]})
    return rows


def load_kjv_jsonl(path: str) -> Dict[str, str]:
    kjv: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            ref = (obj.get("ref") or "").strip()
            txt = (obj.get("kjv") or obj.get("text") or "").strip()
            if not ref or not txt:
                continue
            kjv[ref] = txt
    return kjv


def read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def normalize_space(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def strip_internal_linebreaks(s: str) -> str:
    # Force single-line verse output
    return normalize_space(s.replace("\r", " ").replace("\n", " "))


def similarity_guard(original: str, revised: str) -> Tuple[bool, str]:
    o = normalize_space(original)
    r = normalize_space(revised)

    if not r:
        return False, "empty_output"

    # Reject “commentary”
    if r.startswith("#") or r.lower().startswith(("note:", "commentary:", "explanation:", "translator")):
        return False, "commentary_or_heading"

    # Reject if it starts like “23 …”
    if re.match(r"^\d+\s", r):
        return False, "added_verse_number"

    # Basic length sanity (don’t let it collapse or balloon)
    if len(o) >= 20:
        ratio = len(r) / max(1, len(o))
        if ratio < 0.60:
            return False, "too_short"
        if ratio > 1.60:
            return False, "too_long"

    return True, "ok"


# -------------------------
# Deterministic enforcement
# -------------------------

def enforce_between_from(text: str) -> str:
    text = re.sub(r"\bseparated between\b", "separated", text, flags=re.IGNORECASE)
    text = re.sub(r"\bdivid(e|ing) between\b", r"divid\1", text, flags=re.IGNORECASE)
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
    # Keep "Lord GOD" as-is (Adonai YHWH style)
    text = re.sub(r"\bthe Lord GOD\b", "the Lord GOD", text)

    # Normalize "Lord God" -> "LORD God"
    text = re.sub(r"\bLord God\b", "LORD God", text)

    # Narrative formula "And the Lord said" -> "And the LORD said"
    text = re.sub(r"\bAnd the Lord said\b", "And the LORD said", text)

    # General: "the Lord" -> "the LORD" (but avoid changing "Lord GOD")
    text = re.sub(r"\bthe Lord\b(?!\s+GOD\b)", "the LORD", text)

    # Keep "angel of the LORD" consistent
    text = re.sub(r"\bangel of the Lord\b", "angel of the LORD", text)

    return text


def enforce_compound_numbers(text: str) -> str:
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


def enforce_reverential_pronouns_minimal(text: str) -> str:
    """
    Minimal, conservative pronoun-cap rule:
    Capitalize He/Him/His/Himself only when God/LORD appears in the same verse.
    (You later adopted KJV normalization for full correctness; keep this minimal.)
    """
    if not re.search(r"\b(God|LORD|Lord GOD|the LORD)\b", text):
        return text

    text = re.sub(r"\bhe\b", "He", text)
    text = re.sub(r"\bhim\b", "Him", text)
    text = re.sub(r"\bhis\b", "His", text)
    text = re.sub(r"\bhimself\b", "Himself", text)
    return text


def validate_enforcement(text: str) -> None:
    if "angel of the LORD" in text and "angel of the Lord" in text:
        raise ValueError("Mixed 'angel of the LORD' and 'angel of the Lord' after enforcement")


def apply_enforcement(text: str) -> str:
    text = enforce_lord_caps(text)
    text = enforce_between_from(text)
    text = enforce_compound_numbers(text)
    text = enforce_reverential_pronouns_minimal(text)
    validate_enforcement(text)
    return text


# -------------------------
# Phase-3 heritage mode
# -------------------------

def build_system_prompt(charter_text: str) -> str:
    return (
        charter_text
        + "\n\n"
        + "GLOBAL OUTPUT RULES\n"
        + "- Do NOT add verse numbers, headings, or commentary.\n"
        + "- Output ONLY the revised verse text.\n"
        + "- Output must be ONE line (no internal line breaks).\n"
    )


def build_user_prompt(ref: str, original: str, heritage_kjv: bool, kjv_text: Optional[str]) -> str:
    if heritage_kjv and kjv_text:
        return (
            f"REFERENCE: {ref}\n"
            f"Task: Revise the English verse for oral readability, cadence, and recognizability.\n"
            f"Constraints: do NOT change meaning or theology. Do NOT add commentary.\n"
            f"Style: Preserve or recover widely recognized historic phrasing when faithful.\n"
            f"Avoid archaic pronouns (thou/thee/thy). Mild older phrasing is acceptable if remembered.\n"
            f"Output only the revised verse text as ONE line.\n\n"
            f"KJV CADENCE ANCHOR (public domain):\n{kjv_text}\n\n"
            f"CURRENT VERSE TO REVISE:\n{original}\n"
        )

    return (
        f"REFERENCE: {ref}\n"
        f"Task: Revise the following English verse for cadence and beauty.\n"
        f"Constraints: do NOT change meaning or theology. Do NOT add commentary.\n"
        f"Output only the revised verse text as ONE line.\n\n"
        f"CURRENT VERSE:\n{original}\n"
    )


def main():
    parser = argparse.ArgumentParser(description="Polish pass (Phase 2/3 via charter + targets)")
    parser.add_argument("--in", dest="inp", required=True)
    parser.add_argument("--out", dest="out", required=True)
    parser.add_argument("--targets", required=True)
    parser.add_argument("--charter", required=True)

    parser.add_argument("--model", default="gpt-5.1")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--max_output_tokens", type=int, default=300)

    parser.add_argument("--enforce", action="store_true", help="Apply deterministic enforcement rules")
    parser.add_argument("--enforce_only", action="store_true", help="Do not call model; only enforcement")

    # Phase-3 heritage options
    parser.add_argument("--kjv_path", default="", help="Path to KJV JSONL (ref->kjv)")
    parser.add_argument("--heritage_kjv", action="store_true", help="Use KJV verse as cadence anchor in prompt")

    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not args.enforce_only and not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing or empty")

    targets = load_targets_jsonl(args.targets)
    rows = load_phase_jsonl(args.inp)
    charter_text = read_text_file(args.charter)
    system_prompt = build_system_prompt(charter_text)

    kjv_map: Dict[str, str] = {}
    if args.heritage_kjv:
        if not args.kjv_path:
            raise ValueError("--heritage_kjv requires --kjv_path")
        if not os.path.isfile(args.kjv_path):
            raise FileNotFoundError(f"KJV file not found: {args.kjv_path}")
        if os.path.getsize(args.kjv_path) == 0:
            raise ValueError(f"KJV file is empty (0 bytes): {args.kjv_path}")
        kjv_map = load_kjv_jsonl(args.kjv_path)

    client = OpenAI(api_key=api_key) if not args.enforce_only else None

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    changed = 0
    blocked = 0
    missing_kjv = 0

    with open(args.out, "w", encoding="utf-8") as fout:
        for row in rows:
            ref = row["ref"]
            original = strip_internal_linebreaks(row["translation"])

            # Default: carry original through unchanged
            out_text = original

            # Decide whether to run model for this verse
            should_polish = (ref in targets) and (not args.enforce_only)

            if should_polish:
                kjv_text = None
                if args.heritage_kjv:
                    kjv_text = kjv_map.get(ref)
                    if not kjv_text:
                        # Not fatal; we can still polish without KJV anchor
                        missing_kjv += 1

                user_prompt = build_user_prompt(ref, original, args.heritage_kjv, kjv_text)

                response = client.responses.create(
                    model=args.model,
                    input=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=args.temperature,
                    max_output_tokens=args.max_output_tokens,
                )

                revised = strip_internal_linebreaks(response.output_text.strip())

                if args.enforce:
                    revised = apply_enforcement(revised)

                ok, reason = similarity_guard(original, revised)
                if not ok:
                    print(f"WARNING: Guard blocked {ref}: {reason}")
                    revised = original
                    blocked += 1
                else:
                    if normalize_space(revised) != normalize_space(original):
                        changed += 1

                out_text = revised

            else:
                # no model call; only enforcement optionally
                if args.enforce:
                    out_text = apply_enforcement(out_text)

            fout.write(json.dumps({"ref": ref, "translation": out_text}, ensure_ascii=False) + "\n")

            if args.sleep:
                time.sleep(args.sleep)

    print(f"Polish complete. Changed: {changed} | Guard-blocked: {blocked} | Missing KJV refs: {missing_kjv}")


if __name__ == "__main__":
    main()
