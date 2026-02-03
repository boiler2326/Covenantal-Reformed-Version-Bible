#!/usr/bin/env python3
"""
Phase-2 polish pass:
- Uses an LLM (optional, targeted by refs in targets.jsonl) to revise English for cadence/beauty
- Optionally applies deterministic enforcement rules (LORD caps, between/from cleanup, numbers, reverential pronouns)

Key design goals:
- Book-agnostic: works for any book JSONL in the same {ref, translation} format
- Safe defaults: enforcement is conservative to avoid mis-capitalizing human pronouns (e.g., Moses/Pharaoh in Exodus)
- Deterministic enforcement runs on ALL verses when --enforce is set (targets or not)
"""

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple

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
    """
    # Keep "Lord GOD" as-is (do not turn into LORD GOD)
    text = re.sub(r"\bthe Lord GOD\b", "the Lord GOD", text)

    # Common narrative formulas
    text = re.sub(r"\bAnd the Lord said\b", "And the LORD said", text)
    text = re.sub(r"\bAnd the Lord spoke\b", "And the LORD spoke", text)

    # General: "the Lord" -> "the LORD" (avoid changing "Lord GOD")
    text = re.sub(r"\bthe Lord\b(?!\s+GOD\b)", "the LORD", text)

    # Keep "angel of the LORD" consistent
    text = re.sub(r"\bangel of the Lord\b", "angel of the LORD", text)

    # Normalize "Lord God" -> "LORD God" (YHWH Elohim pattern)
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
    
HUMAN_SUBJECTS_RE = re.compile(
    r"\b(Moses|Aaron|Pharaoh|Joshua|Bezalel|Oholiab|Israel|the people|the man|the woman)\b"
)

DIVINE_MARKERS_RE = re.compile(r"\b(God|the LORD|LORD|Yahweh|Lord GOD)\b")

def enforce_pronoun_false_positive_killers(text: str) -> str:
    """
    Remove known false positives where pronouns refer to humans even though the verse contains LORD/God.
    This should run BEFORE reverential capitalization.
    """

    # Common narrative formula: "the LORD commanded him; so he did"
    text = re.sub(r"\b(the LORD|LORD|Yahweh)\s+commanded\s+Him\b", r"\1 commanded him", text)
    text = re.sub(r"\b(the LORD|LORD|Yahweh)\s+had\s+commanded\s+Him\b", r"\1 had commanded him", text)

    # If we now have "...commanded him; so He did" -> force "so he did"
    text = re.sub(r"\b(commanded\s+him;)\s+so\s+He\s+did\b", r"\1 so he did", text)

    return text


# --- Improved pronoun heuristic (conservative) ---

_PREP_OBJECT_BAN = {
    "to", "from", "with", "by", "for", "at", "in", "on", "upon", "before",
    "after", "against", "toward", "towards", "into", "unto", "over", "under",
    "behind", "beside", "among", "between", "within", "without", "near"
}

# Verbs commonly used when God addresses humans; the object pronoun should remain lowercase:
_ADDRESS_VERBS = {
    "said", "spoke", "called", "cried", "appeared", "showed", "commanded",
    "charged", "sent", "told", "revealed", "made", "gave", "brought"
}

# Human names that frequently appear in Exodus narrative where mis-capitalization is harmful.
# (This list is intentionally limited; it is a "do-no-harm" list.)
_HUMAN_HINTS = {
    "Moses", "Aaron", "Pharaoh", "Israel", "Egypt", "Midian", "Jethro",
    "Joshua", "Miriam", "Hur"
}

# Possessions/attributes very likely divine when paired with "His" in a verse that clearly refers to God.
# This is a whitelist to avoid "His heart" (often Pharaoh) and similar errors.
_DIVINE_POSSESSIONS = {
    "name", "glory", "covenant", "statutes", "commandments", "word", "words",
    "law", "ways", "hand", "hands", "face", "presence", "spirit", "mercy",
    "wrath", "anger", "voice", "mouth", "throne", "kingdom", "holiness", "power",
    "love", "charity"
}

# Patterns that strongly indicate divine first-person speech in the same verse.
_DIVINE_SPEECH_MARKERS = [
    r"\bAnd the LORD said\b",
    r"\bAnd the LORD spoke\b",
    r"\bThus says the LORD\b",
    r"\bsays the LORD\b",
    r"\bGod said\b",
    r"\bthe LORD said\b",
    r"\bthe LORD spoke\b",
]


_WORD_RE = re.compile(r"[A-Za-z']+|[0-9]+|[^\w\s]", re.UNICODE)


def _tokenize(text: str) -> List[str]:
    return _WORD_RE.findall(text)


def _is_divine_context(text: str) -> bool:
    # Explicit divine names in the verse
    if re.search(r"\b(the LORD|LORD God|Lord GOD|God)\b", text):
        return True
    return False


def _has_divine_speech_marker(text: str) -> bool:
    for pat in _DIVINE_SPEECH_MARKERS:
        if re.search(pat, text):
            return True
    return False


def _nearest_antecedent_is_divine(tokens: List[str], idx: int, window: int = 12) -> bool:
    """
    Decide antecedent directionally by scanning backward a short window.
    Conservative: if we see a human hint AFTER the nearest divine hint, we assume human.
    """
    start = max(0, idx - window)
    back = tokens[start:idx]

    # Track nearest divine mention index and nearest human mention index within the window
    nearest_divine = -1
    nearest_human = -1

    for j, tok in enumerate(back):
        # normalize apostrophes? not needed
        if tok in _HUMAN_HINTS:
            nearest_human = j
        if tok in ("God", "LORD", "GOD") or (tok == "LORD" and j >= 1):
            nearest_divine = j
        # also catch "the LORD" as two tokens: "the", "LORD"
        if tok == "LORD":
            nearest_divine = j

    # If there's a human mention closer than the nearest divine mention, avoid capitalization
    if nearest_human != -1 and (nearest_divine == -1 or nearest_human > nearest_divine):
        return False

    return nearest_divine != -1


def enforce_reverential_pronouns(text: str) -> str:
    """
    Improved, conservative reverential pronoun capitalization.

    Goals:
    - Capitalize He/Him/His/Himself only when antecedent is clearly God/LORD in the same verse/clause.
    - Avoid false positives when God is speaking to/appearing to a human (e.g., "called to him").
    - Add optional first-person caps (My/Me/Mine/Myself) ONLY when a divine speech marker is present.

    This is intentionally conservative: it will miss some legitimate divine pronouns rather than mis-capitalizing humans.
    """
    if not _is_divine_context(text):
        return text

    tokens = _tokenize(text)
    if not tokens:
        return text

    divine_speech = _has_divine_speech_marker(text)

    def prev_word(i: int) -> str:
        # previous alphabetic token lowercased
        j = i - 1
        while j >= 0:
            if re.match(r"[A-Za-z']+$", tokens[j]):
                return tokens[j].lower()
            j -= 1
        return ""

    def next_word(i: int) -> str:
        j = i + 1
        while j < len(tokens):
            if re.match(r"[A-Za-z']+$", tokens[j]):
                return tokens[j].lower()
            j += 1
        return ""

    def any_address_verb_near(i: int, lookback: int = 4) -> bool:
        start = max(0, i - lookback)
        for j in range(start, i):
            if re.match(r"[A-Za-z']+$", tokens[j]) and tokens[j].lower() in _ADDRESS_VERBS:
                return True
        return False

    out = tokens[:]
    for i, tok in enumerate(tokens):
        low = tok.lower()

        # First-person divine pronouns: only if divine speech marker present (very conservative)
        if low in {"my", "me", "mine", "myself"}:
            if divine_speech:
                out[i] = tok[:1].upper() + tok[1:] if tok.isalpha() else tok
            continue

        # Third-person pronouns
        if low not in {"he", "him", "his", "himself"}:
            continue

        # If already capitalized, keep unless clearly human (we won't downcase here to avoid unintended edits)
        # We'll only *upcase* when confident.

        # Ban common "God addressing human" object patterns: "... said to him", "... called to him", etc.
        p = prev_word(i)
        if low in {"him", "himself"} and p in _PREP_OBJECT_BAN:
            # if preceded by an address verb, almost certainly human object: "said to him"
            if any_address_verb_near(i, lookback=5):
                continue

        # Possessive "his": only capitalize if the following noun is a divine possession AND antecedent is divine
        if low == "his":
            poss = next_word(i)
            if poss not in _DIVINE_POSSESSIONS:
                continue
            if _nearest_antecedent_is_divine(tokens, i, window=14):
                out[i] = "His"
            continue

        # Subject/object pronouns "he/him/himself"
        if not _nearest_antecedent_is_divine(tokens, i, window=14):
            continue

        # Additional safety: if a human hint appears very near immediately before pronoun, skip
        # (helps cases like "... Moses ... he ...")
        near_start = max(0, i - 6)
        if any(t in _HUMAN_HINTS for t in tokens[near_start:i]):
            continue

        # If it's "he" at the start of a clause, still conservative; only capitalize if divine speech marker present
        if low == "he" and i < 2 and not divine_speech:
            continue

        # Approve capitalization
        if low == "he":
            out[i] = "He"
        elif low == "him":
            out[i] = "Him"
        elif low == "himself":
            out[i] = "Himself"

    return "".join(_rejoin_tokens(out))


def _rejoin_tokens(tokens: List[str]) -> List[str]:
    """
    Rejoin tokens with sane spacing:
    - Space between words
    - No space before punctuation like ,.;:!?)]
    - No space after opening punctuation like ([{
    - Preserve apostrophes inside tokens as they were tokenized
    """
    out: List[str] = []
    for i, tok in enumerate(tokens):
        if i == 0:
            out.append(tok)
            continue

        prev = tokens[i - 1]

        if re.match(r"[^\w\s]", tok):  # punctuation
            # no leading space before punctuation
            out.append(tok)
        elif re.match(r"[^\w\s]", prev) and prev in "([{\u201c\u2018":  # opening punct, quotes
            out.append(tok)
        else:
            out.append(" " + tok)

    return out


def validate_enforcement(text: str) -> None:
    # Example hard-fail: mixed angel phrase variants
    if "angel of the LORD" in text and "angel of the Lord" in text:
        raise ValueError("Mixed 'angel of the LORD' and 'angel of the Lord' after enforcement")


def apply_enforcement(text: str) -> str:
    text = enforce_lord_caps(text)
    text = enforce_between_from(text)
    text = enforce_compound_numbers(text)
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

    with open(args.out, "w", encoding="utf-8") as fout:
        for row in phase1_rows:
            ref = row["ref"]
            original = row["translation"]

            # Start from the original
            out_text = original

            # If in targets, run LLM polish
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
                    revised = apply_enforcement(revised)

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
                    enforced = apply_enforcement(out_text)
                    if normalize_space(enforced) != normalize_space(out_text):
                        changed += 1
                    out_text = enforced

            fout.write(json.dumps({"ref": ref, "translation": out_text}, ensure_ascii=False) + "\n")

            if args.sleep:
                time.sleep(args.sleep)

    print(
        f"Phase-2 complete. LLM-used: {llm_used} | Changed: {changed} | Guard-blocked: {blocked}"
    )


if __name__ == "__main__":
    main()
