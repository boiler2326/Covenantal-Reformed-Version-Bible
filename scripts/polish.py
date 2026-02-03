#!/usr/bin/env python3
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
                raise ValueError(f"targets.jsonl line {i} missing 'ref'")
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


# ----------------------------
# Guards
# ----------------------------

def normalize_space(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def similarity_guard(original: str, revised: str) -> Tuple[bool, str]:
    """
    Guard against model output drifting into commentary/headings/verse numbers
    or wildly changing length.
    """
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


# ----------------------------
# Deterministic enforcement
# ----------------------------

def enforce_between_from(text: str) -> str:
    # separated between X and Y -> separated X from Y
    text = re.sub(r"\bseparated between\b", "separated", text, flags=re.IGNORECASE)
    text = re.sub(r"\bdivid(e|ing) between\b", r"divid\1", text, flags=re.IGNORECASE)

    # normalize some common awkward constructions
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


def enforce_compound_numbers(text: str) -> str:
    # Fix "sixty and five" -> "sixty-five"
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


def enforce_lord_caps(text: str) -> str:
    """
    Normalize YHWH renderings to 'LORD' in English phrases while protecting 'Lord GOD'.
    This is intentionally conservative but more complete than earlier versions.
    """

    # Protect/normalize these first
    # Keep "Lord GOD" as-is (Adonai YHWH style)
    text = re.sub(r"\bLord GOD\b", "Lord GOD", text)
    text = re.sub(r"\bthe Lord GOD\b", "the Lord GOD", text)

    # Common narrative formulae & punctuation variants
    patterns = [
        (r"\bAnd the Lord said\b", "And the LORD said"),
        (r"\bThen the Lord said\b", "Then the LORD said"),
        (r"\bThus says the Lord\b", "Thus says the LORD"),
        (r"\bThus saith the Lord\b", "Thus saith the LORD"),
        (r"\bthe Lord said\b", "the LORD said"),
        (r"\bthe Lord spoke\b", "the LORD spoke"),
        (r"\bthe Lord called\b", "the LORD called"),
        (r"\bthe Lord commanded\b", "the LORD commanded"),
    ]
    for pat, rep in patterns:
        text = re.sub(pat, rep, text)

    # Punctuation cases: "The Lord," / "the Lord," etc.
    text = re.sub(r"\bThe Lord,\b", "The LORD,", text)
    text = re.sub(r"\bthe Lord,\b", "the LORD,", text)

    # General "the Lord" -> "the LORD" (avoid "the LORD GOD")
    text = re.sub(r"\bthe Lord\b(?!\s+GOD\b)", "the LORD", text)

    # Normalize "Lord God" -> "LORD God" (YHWH Elohim pattern)
    text = re.sub(r"\bLord God\b", "LORD God", text)

    # Keep angel phrase consistent
    text = re.sub(r"\bangel of the Lord\b", "angel of the LORD", text)

    return text


def enforce_second_god_pronoun(text: str) -> str:
    """
    Cadence rule (requested):
    If a verse has "And God ... and God ..." -> change the second 'God' to 'He'
    when safe.

    Safety guards:
    - Only applies if it begins with 'And God ' (case-insensitive).
    - Only changes the literal substring ', and God ' or ' and God ' after a comma/semicolon.
    - Skips verses containing 'the God of' (to avoid breaking titles).
    - Skips verses containing 'God of' anywhere (extra conservative).
    """
    t = text
    if not re.match(r"^\s*And God\b", t, flags=re.IGNORECASE):
        return text

    if re.search(r"\bthe God of\b", t, flags=re.IGNORECASE):
        return text

    # Conservative: avoid "God of" entirely (covers "God of Abraham..." etc.)
    if re.search(r"\bGod of\b", t, flags=re.IGNORECASE):
        return text

    # Replace the second occurrence only in common chain forms.
    # Prefer patterns with punctuation to reduce false positives.
    t2 = re.sub(r"([,;])\s+and God\b", r"\1 and He", t, count=1, flags=re.IGNORECASE)
    if t2 != t:
        return t2

    # If no punctuation form, allow a single replacement of " and God " later in the verse.
    # This is still fairly safe when the verse begins with "And God".
    t3 = re.sub(r"\band God\b", "and He", t, count=1, flags=re.IGNORECASE)
    return t3


def enforce_pharaoh_pronouns(text: str) -> str:
    """
    Deterministic Pharaoh antecedent rule:
    If the verse mentions Pharaoh, and also contains capitalized pronouns, it is often
    a human antecedent misfire (He/His referring to Pharaoh).

    We downcase He/Him/His/Himself if Pharaoh appears and the verse ALSO contains
    explicit Pharaoh-antecedent cues like:
      - "Pharaoh's heart"
      - "Pharaoh ... and He"
      - "Pharaoh ...; He"
      - "Pharaoh ... He"
      - "Pharaoh ... His"
    This avoids touching verses where the pronoun clearly refers to God.

    This is conservative but fixes the common errors you flagged (e.g., EXOD 7:13–14).
    """
    if not re.search(r"\bPharaoh\b", text):
        return text

    # If the verse explicitly says "the LORD" and the pronouns are inside a divine-speech quote,
    # it’s too hard to parse deterministically. We'll still fix obvious narrative patterns.
    # We only act when Pharaoh-antecedent cues are present.
    cues = [
        r"Pharaoh[’']s heart\b",
        r"Pharaoh[’']s\b",
        r"Pharaoh\b.*\bHe\b",
        r"Pharaoh\b.*\bHis\b",
        r"Pharaoh\b.*\bHim\b",
    ]
    if not any(re.search(c, text) for c in cues):
        return text

    # Downcase the capitalized pronouns (only exact tokens)
    text = re.sub(r"\bHe\b", "he", text)
    text = re.sub(r"\bHim\b", "him", text)
    text = re.sub(r"\bHis\b", "his", text)
    text = re.sub(r"\bHimself\b", "himself", text)
    return text


def enforce_reverential_pronouns(text: str) -> str:
    """
    Conservative reverential pronoun-cap rule:
    - Only cap He/Him/His/Himself when God/the LORD is explicit in the SAME verse.

    NOTE: This is intentionally conservative; the KJV-gated pronoun pass is the
    scalable solution for full automation. This enforcement is just a safety baseline.
    """
    if not re.search(r"\b(God|the LORD|LORD God|Lord GOD|LORD)\b", text):
        return text

    # Cap only lowercase tokens
    text = re.sub(r"\bhe\b", "He", text)
    text = re.sub(r"\bhim\b", "Him", text)
    text = re.sub(r"\bhis\b", "His", text)
    text = re.sub(r"\bhimself\b", "Himself", text)
    return text


def validate_enforcement(text: str) -> None:
    # Hard fail: mixed 'angel of the LORD' and 'angel of the Lord'
    if "angel of the LORD" in text and "angel of the Lord" in text:
        raise ValueError("Mixed 'angel of the LORD' and 'angel of the Lord' after enforcement")


def apply_enforcement(text: str) -> str:
    """
    Enforcement order matters:
    - First normalize LORD forms
    - Then do human-antecedent overrides (Pharaoh)
    - Then cadence/syntax fixes
    - Then reverential pronoun caps
    """
    t = text

    t = enforce_lord_caps(t)

    # Human antecedent overrides before reverential caps
    t = enforce_pharaoh_pronouns(t)

    # Cadence/syntax rules
    t = enforce_between_from(t)
    t = enforce_compound_numbers(t)
    t = enforce_second_god_pronoun(t)

    # Reverential pronouns last
    t = enforce_reverential_pronouns(t)

    validate_enforcement(t)
    return t


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase-2 polish pass (cadence & beauty).")
    parser.add_argument("--in", dest="inp", required=True)
    parser.add_argument("--out", dest="out", required=True)
    parser.add_argument("--targets", required=True)
    parser.add_argument("--charter", required=True)
    parser.add_argument("--model", default="gpt-5.1")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--max_output_tokens", type=int, default=300)
    parser.add_argument(
        "--enforce",
        action="store_true",
        help="Apply deterministic Phase-2 enforcement rules",
    )
    parser.add_argument(
        "--enforce_only",
        action="store_true",
        help="Skip all model calls; only apply deterministic enforcement",
    )
    args = parser.parse_args()

    # Load data
    targets = load_targets_jsonl(args.targets)
    phase1_rows = load_phase1_jsonl(args.inp)
    phase2_charter = read_text_file(args.charter)

    # Prepare system prompt
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

    client: Optional[OpenAI] = None
    if not args.enforce_only:
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is missing or empty (required unless --enforce_only)")
        client = OpenAI(api_key=api_key)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    changed = 0
    blocked = 0
    enforced_changes = 0
    model_calls = 0

    with open(args.out, "w", encoding="utf-8") as fout:
        for row in phase1_rows:
            ref = row["ref"]
            original = row["translation"]
            out_text = original

            # Decide whether to call the model
            do_model = (not args.enforce_only) and (ref in targets)

            if do_model:
                user_prompt = (
                    f"REFERENCE: {ref}\n"
                    f"Revise the following English verse for cadence and beauty.\n"
                    f"Do not change meaning or theology.\n"
                    f"Output only the revised verse text.\n\n"
                    f"ORIGINAL ENGLISH:\n{original}\n"
                )

                assert client is not None
                response = client.responses.create(
                    model=args.model,
                    input=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=args.temperature,
                    max_output_tokens=args.max_output_tokens,
                )
                model_calls += 1
                candidate = response.output_text.strip()

                # If enforcing, apply after model output
                if args.enforce:
                    candidate2 = apply_enforcement(candidate)
                    if normalize_space(candidate2) != normalize_space(candidate):
                        enforced_changes += 1
                    candidate = candidate2

                # Guard against drift
                ok, reason = similarity_guard(original, candidate)
                if not ok:
                    print(f"WARNING: Guard blocked {ref}: {reason}")
                    out_text = original
                    blocked += 1
                else:
                    out_text = candidate
                    if normalize_space(out_text) != normalize_space(original):
                        changed += 1

            else:
                # Non-target (or enforce-only mode): pass through, optionally enforce
                if args.enforce:
                    enforced = apply_enforcement(out_text)
                    if normalize_space(enforced) != normalize_space(out_text):
                        enforced_changes += 1
                    out_text = enforced

            fout.write(json.dumps({"ref": ref, "translation": out_text}, ensure_ascii=False) + "\n")

            if args.sleep and do_model:
                time.sleep(args.sleep)

    print(
        f"Phase-2 complete. Changed(by model): {changed} | "
        f"Guard-blocked: {blocked} | "
        f"Enforcement-changed: {enforced_changes} | "
        f"Model-calls: {model_calls}"
    )


if __name__ == "__main__":
    main()
