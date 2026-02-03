#!/usr/bin/env python3
import argparse
import json
import os
import re
import time
from typing import Dict, List, Set, Tuple

from openai import OpenAI


# ----------------------------
# IO helpers
# ----------------------------

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


# ----------------------------
# Guards
# ----------------------------

def normalize_space(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def similarity_guard(original: str, revised: str) -> Tuple[bool, str]:
    o = normalize_space(original)
    r = normalize_space(revised)

    if not r:
        return False, "empty_output"

    if r.startswith("#") or r.lower().startswith(
        ("note:", "commentary:", "explanation:", "translator")
    ):
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

    # normalize "to divide between the day and between the night" -> "to divide the day from the night"
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


# ----------------------------
# Reverential pronoun logic (improved)
# ----------------------------

DIVINE_SIGNAL_RE = re.compile(
    r"\b("
    r"the LORD|LORD God|Lord GOD|LORD\b|God\b|Almighty\b|Most High\b|"
    r"I AM\b|Holy One\b"
    r")\b"
)

# If these appear as explicit local antecedents, pronoun caps must NOT apply to them.
# (We keep this conservative; it’s a “don’t capitalise for humans” list.)
HUMAN_ANTECEDENT_RE = re.compile(
    r"\b("
    r"Moses|Aaron|Pharaoh|Israel|Egyptians?|Hebrews?|people|man|woman|"
    r"son|daughter|brother|father|mother|servant|priest|king|"
    r"Zipporah|Jethro|Joshua|Amram|Miriam"
    r")\b"
)

# Object-verb patterns that almost always take a human object in Exodus narrative,
# even inside a verse that also contains “the LORD”.
# Example: “the LORD … commanded him” (object is human).
HUMAN_OBJECT_VERB_PATTERNS = [
    # (verb phrase, object pronoun to de-cap)
    r"\bmet Him\b",
    r"\bsought to put Him to death\b",
    r"\bkissed Him\b",
    r"\bsent Him\b",
    r"\bcommanded Him\b",
    r"\bspoke to Him\b",
    r"\bsaid to Him\b",
    r"\btold Him\b",
    r"\bcalled to Him\b",
    r"\btook Him\b",
    r"\bbrought Him\b",
]

# Subject patterns that indicate the “He” in the clause is a human (not God),
# even if “the LORD” appears earlier in the verse.
HUMAN_SUBJECT_CUES = re.compile(
    r"\b(he|He)\s+("
    r"went|came|returned|ran|met|kissed|took|lifted|stretched|"
    r"stood|sat|spoke|said|answered|told|called|bowed|"
    r"built|made|wrote|smote|struck"
    r")\b"
)

def _decap_pronoun_word(match: re.Match) -> str:
    # Only decap exactly these forms; leave “I/Me/My” alone.
    w = match.group(0)
    mapping = {"He": "he", "Him": "him", "His": "his", "Himself": "himself"}
    return mapping.get(w, w)

def enforce_reverential_pronouns(text: str) -> str:
    """
    Two-stage approach:

    Stage A (cap): If the verse clearly signals God/the LORD, capitalize He/Him/His/Himself.
    Stage B (decap exceptions): If context shows the pronoun refers to a human,
    deterministically lower it back.
    """

    # Stage A: only cap when we see a divine signal in the verse
    if DIVINE_SIGNAL_RE.search(text):
        # Cap standalone pronouns; use word boundaries to avoid “the” etc.
        text = re.sub(r"\bhe\b", "He", text)
        text = re.sub(r"\bhim\b", "Him", text)
        text = re.sub(r"\bhis\b", "His", text)
        text = re.sub(r"\bhimself\b", "Himself", text)

    # Stage B: exceptions / rollbacks for humans
    # 1) If explicit human antecedent is present AND we see “said to Him / spoke to Him / told Him” etc,
    #    it is almost certainly human-directed. Roll those objects back.
    if HUMAN_ANTECEDENT_RE.search(text):
        for pat in HUMAN_OBJECT_VERB_PATTERNS:
            text = re.sub(pat, lambda m: _decap_pronoun_word(m), text)

        # Specifically fix common “the LORD … sent Him / commanded Him” where Him is Moses/Aaron etc.
        text = re.sub(r"\b(sent|commanded)\s+Him\b", r"\1 him", text)
        text = re.sub(r"\b(spoke|said|told|called)\s+to\s+Him\b", r"\1 to him", text)

    # 2) If the clause clearly shows a human subject action, roll back a leading “He …” to “he …”
    #    BUT only when the divine name is not the immediate subject.
    #    This keeps “And the LORD said…” safe, because that doesn’t match HUMAN_SUBJECT_CUES.
    if HUMAN_SUBJECT_CUES.search(text):
        # Lower only the “He” that begins a clause (start or after punctuation / conjunction)
        text = re.sub(r"(^|[;:.!?]\s+|\bAnd\s+)\bHe\b", r"\1he", text)

    # 3) Staff/hand phrases: “staff of God in His hand” in Exodus is Moses’ hand, not God’s.
    text = re.sub(r"\bstaff of God in His hand\b", "staff of God in his hand", text)

    return text


def validate_enforcement(text: str) -> None:
    # Hard fail: mixed LORD/Lord in the same phrase context
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

def main():
    parser = argparse.ArgumentParser(description="Phase-2 polish pass")
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

            # If not a target, optionally enforce-only and pass through
            if ref not in targets:
                out_text = original
                if args.enforce:
                    out_text = apply_enforcement(out_text)
                fout.write(json.dumps({"ref": ref, "translation": out_text}, ensure_ascii=False) + "\n")
                continue

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

            fout.write(json.dumps({"ref": ref, "translation": revised}, ensure_ascii=False) + "\n")

            if args.sleep:
                time.sleep(args.sleep)

    print(f"Phase-2 complete. Changed: {changed} | Guard-blocked: {blocked}")


if __name__ == "__main__":
    main()
