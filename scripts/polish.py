#!/usr/bin/env python3
"""
Phase-2 polish pass

Inputs:
- Phase-1 JSONL: lines of {"ref":"EXOD 1:1","translation":"..."}
- targets.jsonl: lines of {"ref":"EXOD 3:14"} indicating which verses should get LLM "beauty polish"
- phase2_charter.txt: your Phase-2 charter

Behavior:
- If ref is in targets -> call LLM to revise for cadence/beauty; meaning must remain unchanged.
- If ref is NOT in targets -> pass through unchanged.
- If --enforce is set -> apply deterministic enforcement to ALL verses (targets or not).

Deterministic enforcement includes:
- LORD caps normalization
- between/from cleanup
- compound number hyphenation
- improved reverential pronouns (antecedent-aware, conservative)
- explicit false-positive killers ("commanded him; so he did", etc.)
- optional craftsman-run lowercase ("And he made...") when appropriate

Design principle:
- Prefer missing a legitimate divine pronoun cap over incorrectly capitalizing a human pronoun.
"""

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional

from openai import OpenAI


# ----------------------------
# IO helpers
# ----------------------------

def load_targets_jsonl(path: str) -> Set[str]:
    targets: Set[str] = set()
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            ref = (obj.get("ref") or "").strip()
            if not ref:
                raise ValueError(f"targets.jsonl line {line_no} missing 'ref'")
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
                raise ValueError(f"Input jsonl missing keys at line {i}")
            rows.append({"ref": obj["ref"], "translation": obj["translation"]})
    return rows


def read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


# ----------------------------
# Guard / normalization
# ----------------------------

def normalize_space(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def similarity_guard(original: str, revised: str) -> Tuple[bool, str]:
    """
    Guard against model output going off the rails (commentary, headings, verse numbers, huge length changes).
    """
    o = normalize_space(original)
    r = normalize_space(revised)

    if not r:
        return False, "empty_output"

    # No commentary
    if r.startswith("#") or r.lower().startswith(("note:", "commentary:", "explanation:", "translator")):
        return False, "commentary_or_heading"

    # No verse numbers added
    if re.match(r"^\d+\s", r):
        return False, "added_verse_number"

    # Prevent wild shortening/expansion
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

DIVINE_MARKERS_RE = re.compile(r"\b(God|the LORD|LORD|Yahweh|Lord GOD|LORD God)\b")
# Names/subjects that frequently appear where pronouns should remain lowercase.
# Keep limited & conservative; you can add more as needed.
HUMAN_SUBJECTS_RE = re.compile(
    r"\b(Moses|Aaron|Pharaoh|Joshua|Bezalel|Oholiab|Miriam|Hur|Jethro|Israel|the people|the man|the woman)\b"
)

DIVINE_SPEECH_MARKERS_RE = re.compile(
    r"\b(And the LORD said|And the LORD spoke|Thus says the LORD|says the LORD|God said|the LORD said|the LORD spoke)\b"
)

# Possession nouns that are very likely divine when paired with "His"
DIVINE_POSSESSIONS = {
    "name", "glory", "covenant", "statutes", "commandments", "word", "words",
    "law", "ways", "hand", "hands", "face", "presence", "spirit", "mercy",
    "wrath", "anger", "voice", "mouth", "holiness", "power", "love", "charity",
}

ADDRESS_VERBS = {
    "said", "spoke", "called", "cried", "appeared", "showed", "commanded",
    "charged", "sent", "told", "revealed", "gave", "brought", "spared",
}

PREP_OBJECT_BAN = {
    "to", "from", "with", "by", "for", "at", "in", "on", "upon", "before",
    "after", "against", "toward", "towards", "into", "unto", "over", "under",
    "behind", "beside", "among", "between", "within", "without", "near",
}


def enforce_between_from(text: str) -> str:
    # "separated between X and Y" -> "separated X from Y"
    text = re.sub(r"\bseparated between\b", "separated", text, flags=re.IGNORECASE)
    text = re.sub(r"\bdivid(e|ing) between\b", r"divid\1", text, flags=re.IGNORECASE)

    # Normalize common awkward constructions
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
    """
    Conservative LORD handling:
    - Maintain "Lord GOD" (Adonai YHWH style)
    - Normalize "the Lord" -> "the LORD" (YHWH)
    - Normalize "And the Lord said/spoke" -> "And the LORD said/spoke"
    - Normalize "angel of the Lord" -> "angel of the LORD"
    - Normalize "Lord God" -> "LORD God" (YHWH Elohim pattern)
    """
    # Keep "Lord GOD" as-is
    text = re.sub(r"\bthe Lord GOD\b", "the Lord GOD", text)

    # Common narrative formulas
    text = re.sub(r"\bAnd the Lord said\b", "And the LORD said", text)
    text = re.sub(r"\bAnd the Lord spoke\b", "And the LORD spoke", text)

    # General: "the Lord" -> "the LORD" (avoid changing "Lord GOD")
    text = re.sub(r"\bthe Lord\b(?!\s+GOD\b)", "the LORD", text)

    # Keep "angel of the LORD" consistent
    text = re.sub(r"\bangel of the Lord\b", "angel of the LORD", text)

    # Normalize "Lord God" -> "LORD God"
    text = re.sub(r"\bLord God\b", "LORD God", text)

    return text


def enforce_compound_numbers(text: str) -> str:
    # Fix awkward "sixty and five" -> "sixty-five" patterns
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


def enforce_pronoun_false_positive_killers(text: str) -> str:
    """
    Kill known recurring false positives before any reverential capitalization.
    These are "almost always human" when they appear.
    """
    # Narrative formula: the LORD commanded him (object is a human)
    text = re.sub(r"\b(the LORD|LORD|Yahweh)\s+commanded\s+Him\b", r"\1 commanded him", text)
    text = re.sub(r"\b(the LORD|LORD|Yahweh)\s+had\s+commanded\s+Him\b", r"\1 had commanded him", text)

    # If we have "... commanded him; so He did" -> force lowercase
    text = re.sub(r"\b(commanded\s+him;)\s+so\s+He\s+did\b", r"\1 so he did", text)

    return text


def enforce_craftsman_opening_lowercase(text: str) -> str:
    """
    Prevent accidental divine caps in long "he made / he fashioned / he joined" runs.
    If a verse starts with "He made/He joined/He fashioned/He made..." and there is no divine marker,
    prefer "And he ..." for cadence and to avoid reverential false positives.
    """
    if text.startswith(("He made", "He joined", "He fashioned", "He overlaid", "He set")):
        if not DIVINE_MARKERS_RE.search(text):
            # Insert "And " and lowercase the initial He
            return "And he" + text[len("He"):]
    return text


def enforce_reverential_pronouns(text: str) -> str:
    """
    Conservative reverential pronoun capitalization.

    Strategy:
    - Only consider caps if the verse contains a divine marker (God/the LORD/LORD/etc.).
    - For each pronoun, look left within a short window.
        - Require a divine marker in that left-window.
        - If a human subject appears after the last divine marker in that window, do NOT capitalize.
    - Special-case:
        - Do not capitalize object pronouns in "said to him / called to him" patterns (usually human).
        - "his X" only caps if X is a whitelisted divine possession and antecedent is divine.

    Also:
    - First-person My/Me/Mine only caps when a divine speech marker is present.
    """
    if not DIVINE_MARKERS_RE.search(text):
        return text

    # First-person caps only within divine speech
    if DIVINE_SPEECH_MARKERS_RE.search(text):
        # Avoid turning "my" into "My" inside quotes belonging to humans (rare in Torah narrative),
        # but this is still conservative: only do direct word-boundary replacements.
        text = re.sub(r"\bmy\b", "My", text)
        text = re.sub(r"\bme\b", "Me", text)
        text = re.sub(r"\bmine\b", "Mine", text)
        text = re.sub(r"\bmyself\b", "Myself", text)

    # Helper to decide capitalization of third-person pronouns
    def should_cap(pron_match: re.Match) -> bool:
        pron = pron_match.group(0)  # lower pronoun from regex below
        start = pron_match.start()

        # Look back a short window (characters) for antecedent hints
        window_start = max(0, start - 90)
        left = text[window_start:start]

        # Need a divine marker somewhere in the left window
        m_divs = list(DIVINE_MARKERS_RE.finditer(left))
        if not m_divs:
            return False

        # Find the last divine marker and then see if a human subject appears after it
        last_div_end = m_divs[-1].end()
        tail = left[last_div_end:]
        if HUMAN_SUBJECTS_RE.search(tail):
            return False

        # Avoid "said to him / called to him" class: object pronouns are typically humans
        # We'll check the immediate few words before the pronoun in the raw left string.
        # If it ends with "to " / "from " / "with " etc, and an address-verb appears nearby, skip.
        # This catches: "called to him", "said to him", "appeared to him"
        prev_words = re.findall(r"[A-Za-z']+", left.lower())[-6:]
        if pron in ("him", "himself") and prev_words:
            if prev_words[-1] in PREP_OBJECT_BAN:
                # look for an address verb in the previous few words
                if any(w in ADDRESS_VERBS for w in prev_words[:-1]):
                    return False

        # Possessive "his": only cap if next noun is a divine possession
        if pron == "his":
            # Peek right a bit for the next word
            right = text[pron_match.end(): pron_match.end() + 40]
            next_words = re.findall(r"[A-Za-z']+", right)
            if not next_words:
                return False
            noun = next_words[0].lower()
            if noun not in DIVINE_POSSESSIONS:
                return False

        return True

    # Replace only lower-case pronouns, leaving already-capped ones alone
    def repl(m: re.Match) -> str:
        pron = m.group(0)
        if not should_cap(m):
            return pron
        mapping = {"he": "He", "him": "Him", "his": "His", "himself": "Himself"}
        return mapping.get(pron, pron)

    return re.sub(r"\b(he|him|his|himself)\b", repl, text)


def validate_enforcement(text: str) -> None:
    # Example hard-fail: mixed angel phrase variants
    if "angel of the LORD" in text and "angel of the Lord" in text:
        raise ValueError("Mixed 'angel of the LORD' and 'angel of the Lord' after enforcement")


def apply_enforcement(text: str) -> str:
    text = enforce_lord_caps(text)
    text = enforce_between_from(text)
    text = enforce_compound_numbers(text)

    # Kill known false positives before pronoun caps
    text = enforce_pronoun_false_positive_killers(text)

    # Craftsman-run lowercasing (optional but safe)
    text = enforce_craftsman_opening_lowercase(text)

    # Now apply reverential caps conservatively
    text = enforce_reverential_pronouns(text)

    validate_enforcement(text)
    return text


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase-2 polish pass")
    parser.add_argument("--in", dest="inp", required=True, help="Input JSONL (Phase-1 output)")
    parser.add_argument("--out", dest="out", required=True, help="Output JSONL (Phase-2 output)")
    parser.add_argument("--targets", required=True, help="targets.jsonl containing refs to LLM-polish")
    parser.add_argument("--charter", required=True, help="Phase-2 charter text file")
    parser.add_argument("--model", default="gpt-5.1")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--max_output_tokens", type=int, default=300)
    parser.add_argument(
        "--enforce",
        action="store_true",
        help="Apply deterministic Phase-2 enforcement rules to ALL verses",
    )
    parser.add_argument(
        "--debug_enforce",
        action="store_true",
        help="Print refs where enforcement changed the verse text",
    )
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

    Path(os.path.dirname(args.out) or ".").mkdir(parents=True, exist_ok=True)

    changed = 0
    blocked = 0
    llm_used = 0
    enforce_changed = 0

    with open(args.out, "w", encoding="utf-8") as fout:
        for row in phase1_rows:
            ref = row["ref"]
            original = row["translation"]

            out_text = original

            # If ref is in targets, run LLM polish
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

                revised = (response.output_text or "").strip()
                llm_used += 1

                # Apply enforcement to LLM output if enabled
                if args.enforce:
                    before = revised
                    revised = apply_enforcement(revised)
                    if normalize_space(before) != normalize_space(revised):
                        enforce_changed += 1
                        if args.debug_enforce:
                            print(f"ENFORCE CHANGED (LLM): {ref}")

                ok, reason = similarity_guard(original, revised)
                if not ok:
                    print(f"WARNING: Guard blocked {ref}: {reason}")
                    out_text = original
                    blocked += 1
                else:
                    out_text = revised
                    if normalize_space(out_text) != normalize_space(original):
                        changed += 1

            else:
                # Not in targets: just apply enforcement if enabled
                if args.enforce:
                    before = out_text
                    out_text = apply_enforcement(out_text)
                    if normalize_space(before) != normalize_space(out_text):
                        enforce_changed += 1
                        if args.debug_enforce:
                            print(f"ENFORCE CHANGED: {ref}")

            fout.write(json.dumps({"ref": ref, "translation": out_text}, ensure_ascii=False) + "\n")

            if args.sleep:
                time.sleep(args.sleep)

    print(
        f"Phase-2 complete. LLM-used: {llm_used} | Changed: {changed} | "
        f"Enforce-changed: {enforce_changed} | Guard-blocked: {blocked}"
    )


if __name__ == "__main__":
    main()
