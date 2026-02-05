#!/usr/bin/env python3
import json
import csv
import re
from dataclasses import dataclass
from typing import List, Dict, Tuple

# ---------------- CONFIG ----------------
DIVINE_ANCHORS = [
    r"\bLORD\b",
    r"\bGOD\b",
    r"\bO\s+LORD\b",
    r"\bO\s+God\b",
    r"\bGod\b",
    # r"\bLord\b",  # OPTIONAL: uncomment if you want "Lord" to count as an anchor
]

ANCHOR_RE = re.compile("|".join(DIVINE_ANCHORS))

PRONOUN_RE = re.compile(r"\b(he|him|his|you|your|yours|himself)\b")
NEAR_INVOCATION_RE = re.compile(r"\bO\s+(LORD|God)\b", re.IGNORECASE)

# Titles/metaphors we care about. Multi-word phrases supported.
TITLE_PHRASES = [
    "rock",
    "fortress",
    "stronghold",
    "high tower",
    # "salvation" is special (title only in "my/our/his Salvation" etc., NOT "Your salvation" by default)
]

POSSESSIVE_TITLE_RE = re.compile(
    r"\b(my|our|his|their)\s+(rock|fortress|stronghold|high\s+tower)\b",
    re.IGNORECASE,
)

IS_TITLE_RE = re.compile(
    r"\bis\s+(my|our|his|their)\s+(rock|fortress|stronghold|high\s+tower)\b",
    re.IGNORECASE,
)

# Salvation-as-title: only when it functions as a name for God (my/our/his/their salvation)
SALVATION_TITLE_RE = re.compile(r"\b(my|our|his|their)\s+salvation\b", re.IGNORECASE)

@dataclass
class Suggestion:
    ref: str
    original: str
    suggested: str
    reason: str
    confidence: float
    kind: str  # "pronoun" or "title"

def has_anchor(text: str) -> bool:
    return bool(ANCHOR_RE.search(text))

def confidence_for_pronoun(text: str, match_start: int) -> float:
    # Conservative scoring
    score = 0.55
    if has_anchor(text):
        score += 0.25
    window = text[max(0, match_start - 40): match_start + 40]
    if NEAR_INVOCATION_RE.search(window):
        score += 0.15
    if re.match(r"^\s*(God|The LORD|LORD)\b", text):
        score += 0.05
    return min(score, 0.95)

def apply_single_span(text: str, start: int, end: int, replacement: str) -> str:
    return text[:start] + replacement + text[end:]

def cap_phrase(phrase: str) -> str:
    return " ".join(w.capitalize() for w in phrase.split())

def suggest_line(ref: str, text: str) -> List[Suggestion]:
    suggestions: List[Suggestion] = []
    original = text
    working = text

    # --- Pronouns: only propose if verse contains a divine anchor ---
    if has_anchor(working):
        # Iterate matches in the ORIGINAL text to keep indices stable per replacement strategy:
        # We'll rebuild progressively and re-find after each edit by restarting scan.
        while True:
            m = PRONOUN_RE.search(working)
            if not m:
                break
            pron = m.group(1)
            if pron.islower():
                cap = pron[0].upper() + pron[1:]
                conf = confidence_for_pronoun(working, m.start())
                suggested = apply_single_span(working, m.start(), m.end(), cap)
                suggestions.append(Suggestion(
                    ref=ref,
                    original=original,
                    suggested=suggested,
                    reason=f"Divine pronoun '{pron}' not capitalized (anchor present in verse).",
                    confidence=conf,
                    kind="pronoun",
                ))
                working = suggested
            else:
                # Already capitalized: skip this occurrence and continue searching after it
                # Replace with a placeholder mark to avoid infinite loop:
                working = apply_single_span(working, m.start(), m.end(), pron)

    # Reset working back to original for title detection (titles should be detected on original),
    # because pronoun edits shouldn't affect title spans.
    working = original

    # --- Titles (Rock/Fortress/Stronghold/High Tower) as titles ---
    for regex, reason, conf in [
        (IS_TITLE_RE, "Divine title used as identity ('is my/our/his/their ...').", 0.92),
        (POSSESSIVE_TITLE_RE, "Divine title used as possessive title ('my/our/his/their ...').", 0.88),
    ]:
        for m in list(regex.finditer(working)):
            noun = m.group(2).lower()
            noun_cap = cap_phrase(noun)
            start, end = m.span(2)
            suggested = apply_single_span(working, start, end, noun_cap)
            suggestions.append(Suggestion(
                ref=ref,
                original=original,
                suggested=suggested,
                reason=reason,
                confidence=conf,
                kind="title",
            ))
            working = suggested  # allow multiple title caps in same verse

    # --- Salvation as title only in my/our/his/their salvation ---
    for m in list(SALVATION_TITLE_RE.finditer(working)):
        start, end = m.span()
        phrase = working[start:end]
        phrase2 = re.sub(r"\bsalvation\b", "Salvation", phrase, flags=re.IGNORECASE)
        suggested = apply_single_span(working, start, end, phrase2)
        suggestions.append(Suggestion(
            ref=ref,
            original=original,
            suggested=suggested,
            reason="Salvation used as a title (my/our/his/their Salvation).",
            confidence=0.85,
            kind="title",
        ))
        working = suggested

    # Deduplicate identical suggestions per ref
    seen = set()
    uniq: List[Suggestion] = []
    for s in suggestions:
        key = (s.ref, s.suggested)
        if key not in seen:
            seen.add(key)
            uniq.append(s)
    return uniq

def main(in_jsonl: str, out_csv: str):
    all_sugs: List[Suggestion] = []
    with open(in_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            ref = obj.get("ref", "")
            txt = obj.get("translation", "")
            if not ref or not isinstance(txt, str):
                continue
            all_sugs.extend(suggest_line(ref, txt))

    # Output as a review worksheet; "decision" column is empty by default
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["decision", "ref", "confidence", "kind", "reason", "original", "suggested"])
        for s in sorted(all_sugs, key=lambda x: (x.ref, -x.confidence)):
            w.writerow(["", s.ref, f"{s.confidence:.2f}", s.kind, s.reason, s.original, s.suggested])

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python scripts/suggest_caps.py input.jsonl output.csv")
        raise SystemExit(2)
    main(sys.argv[1], sys.argv[2])
