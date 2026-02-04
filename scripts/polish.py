#!/usr/bin/env python3
"""
polish.py

Targets-based polish (Phase 2 / Phase 3) with optional deterministic enforcement.

Key features:
- Reads input book JSONL: {"ref":"PSA 23:1","translation":"..."}
- Reads targets JSONL: {"ref":"PSA 23:1"} (one per line)
- For refs in targets:
    - optionally calls the model using the provided charter
    - can also run deterministic KJV case normalization (Option 2: caps + decaps)
    - can enforce heritage phrase locks
- For refs NOT in targets:
    - copies unchanged (unless --enforce_only is used, which can be applied to all)

Flags:
- --enforce: apply deterministic enforcement rules
- --enforce_only: do not call the model; just apply enforcement (to ALL verses)
- --heritage_kjv + --kjv_path: use KJV verse as cadence anchor in prompt
- --heritage_anchors: JSONL ref->locks[]; ensures locked phrases appear; may fall back to KJV
- --kjv_case: copy casing from KJV for matched tokens (caps + decaps), APPLIES ONLY TO TARGETS
"""

import argparse
import json
import os
import re
import time
from typing import Dict, List, Set, Tuple, Optional

from openai import OpenAI


# -------------------------
# I/O helpers
# -------------------------

def load_targets_jsonl(path: str) -> Set[str]:
    targets: Set[str] = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            ref = (obj.get("ref") or "").strip()
            if not ref:
                raise ValueError("targets.jsonl line missing 'ref'")
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
                raise ValueError(f"Input jsonl missing keys at line {i}")
            rows.append({"ref": obj["ref"], "translation": obj["translation"]})
    return rows


def read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def load_kjv_jsonl(path: str) -> Dict[str, str]:
    kjv: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            ref = (obj.get("ref") or "").strip()
            txt = (obj.get("kjv") or obj.get("text") or "").strip()
            if ref and txt:
                kjv[ref] = txt
    return kjv


def load_heritage_anchors_jsonl(path: str) -> Dict[str, List[str]]:
    """
    Each line:
      {"ref":"PSA 23:1","locks":["I shall not want", ...]}
    Returns: ref -> locks[]
    """
    anchors: Dict[str, List[str]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            ref = (obj.get("ref") or "").strip()
            locks = obj.get("locks") or []
            if not ref:
                raise ValueError(f"heritage_anchors.jsonl missing 'ref' at line {i}")
            if not isinstance(locks, list):
                raise ValueError(f"heritage_anchors.jsonl 'locks' must be a list at line {i}")
            locks = [str(x).strip() for x in locks if str(x).strip()]
            anchors[ref] = locks
    return anchors


# -------------------------
# Text normalization / guards
# -------------------------

def normalize_space(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def strip_internal_linebreaks(s: str) -> str:
    return normalize_space(s.replace("\r", " ").replace("\n", " "))


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
# Deterministic enforcement (Phase 2/3)
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
    text = re.sub(r"\bthe Lord GOD\b", "the Lord GOD", text)
    text = re.sub(r"\bLord God\b", "LORD God", text)
    text = re.sub(r"\bAnd the Lord said\b", "And the LORD said", text)
    text = re.sub(r"\bthe Lord\b(?!\s+GOD\b)", "the LORD", text)
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
    Minimal: capitalize He/Him/His/Himself only when God/LORD appears in the SAME verse.
    (You may later choose to remove this entirely and rely on --kjv_case.)
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
# Heritage helpers (nostalgia locks)
# -------------------------

def dearchaize_pronouns(s: str) -> str:
    """
    Keep KJV cadence but remove archaic 2nd-person pronouns.
    We do NOT attempt broad verb modernization here.
    """
    repl = [
        (r"\bthou\b", "You"),
        (r"\bthee\b", "You"),
        (r"\bthy\b", "Your"),
        (r"\bthine\b", "Your"),
    ]
    out = s
    for pat, rep in repl:
        out = re.sub(pat, rep, out, flags=re.IGNORECASE)
    return out


def enforce_phrase_locks(text: str, locks: List[str], kjv_text: Optional[str]) -> Tuple[str, List[str]]:
    """
    Ensure each lock string appears. If missing, try gentle replacements;
    if still missing and KJV contains all locks, fall back to (de-archaized) KJV text.
    """
    t = text

    def apply_common_subs(t_in: str, lock: str) -> str:
        t_out = t_in

        if lock.lower() == "i shall not want":
            t_out = re.sub(r"\bI shall not lack\b", "I shall not want", t_out)
            t_out = re.sub(r"\bI will not lack\b", "I shall not want", t_out)
            t_out = re.sub(r"\bI do not lack\b", "I shall not want", t_out)

        if lock.lower() == "the valley of the shadow of death":
            t_out = re.sub(r"\bvalley of (the )?deepest shadow\b", "valley of the shadow of death", t_out, flags=re.IGNORECASE)
            t_out = re.sub(r"\bvalley of (the )?darkest shadow\b", "valley of the shadow of death", t_out, flags=re.IGNORECASE)

        if lock.lower() in ("for ever", "for ever."):
            t_out = re.sub(r"\bfor length of days\b", "for ever", t_out, flags=re.IGNORECASE)
            t_out = re.sub(r"\bforever\b", "for ever", t_out, flags=re.IGNORECASE)

        return t_out

    for lock in locks:
        if lock and lock not in t:
            t = apply_common_subs(t, lock)

    missing = [lk for lk in locks if lk and lk not in t]

    if missing and kjv_text:
        kjv_fixed = strip_internal_linebreaks(dearchaize_pronouns(kjv_text))
        if all((lk in kjv_fixed) for lk in locks if lk):
            return kjv_fixed, []

    return t, missing


# -------------------------
# KJV Case normalization (Option 2: caps + decaps) - targets only
# -------------------------

WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+|[^\w\s]", re.UNICODE)

def tokenize(s: str) -> List[str]:
    return WORD_RE.findall(s)

def is_word(tok: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z]+(?:'[A-Za-z]+)?", tok))

def apply_case_pattern(src: str, pattern: str) -> str:
    """
    Apply the casing style of pattern onto src (same letters, new case).
    - If pattern is all upper -> src.upper()
    - If pattern is all lower -> src.lower()
    - If pattern is Title (first upper, rest lower) -> title-ish
    - Else (mixed) -> best-effort per-character where possible
    """
    if pattern.isupper():
        return src.upper()
    if pattern.islower():
        return src.lower()
    if len(pattern) > 1 and pattern[0].isupper() and pattern[1:].islower():
        # Title case
        return src[0].upper() + src[1:].lower()

    # Mixed casing: map letter-by-letter
    out_chars: List[str] = []
    for i, ch in enumerate(src):
        if i < len(pattern) and pattern[i].isalpha():
            out_chars.append(ch.upper() if pattern[i].isupper() else ch.lower())
        else:
            out_chars.append(ch)
    return "".join(out_chars)

def kjv_case_normalize(text: str, kjv_text: str) -> str:
    """
    Copy casing from KJV for matched word tokens.
    Only changes tokens that match (case-insensitive) at the same aligned position
    under a simple greedy alignment.
    """
    a = tokenize(text)
    b = tokenize(kjv_text)

    out = a[:]

    i = 0
    j = 0
    while i < len(a) and j < len(b):
        if is_word(a[i]) and is_word(b[j]) and a[i].lower() == b[j].lower():
            out[i] = apply_case_pattern(a[i], b[j])
            i += 1
            j += 1
            continue

        # Greedy skip: advance the side that looks "more different"
        # If current token is punctuation, move it
        if not is_word(a[i]):
            i += 1
            continue
        if not is_word(b[j]):
            j += 1
            continue

        # both words but not equal -> attempt a small lookahead
        found = False
        for k in range(1, 4):
            if i + k < len(a) and is_word(a[i+k]) and a[i+k].lower() == b[j].lower():
                i += k
                found = True
                break
            if j + k < len(b) and is_word(b[j+k]) and a[i].lower() == b[j+k].lower():
                j += k
                found = True
                break
        if not found:
            i += 1
            j += 1

    return strip_internal_linebreaks("".join(out) if all(len(t) == 1 and not is_word(t) for t in out) else " ".join(out).replace("  ", " "))


def join_tokens(tokens: List[str]) -> str:
    """
    Join tokens into a readable string:
    - No spaces before punctuation like ,.;:!?)]
    - No spaces after opening punctuation like ([{
    """
    s = ""
    for t in tokens:
        if not s:
            s = t
            continue
        if re.fullmatch(r"[,\.;:\?!\)\]\}]", t):
            s += t
        elif re.fullmatch(r"[\(\[\{]", t):
            s += " " + t
        else:
            # default
            if s.endswith(("(", "[", "{", "“", '"', "‘", "'")):
                s += t
            else:
                s += " " + t
    return strip_internal_linebreaks(s)


def kjv_case_normalize_v2(text: str, kjv_text: str) -> str:
    a = tokenize(text)
    b = tokenize(kjv_text)

    out = a[:]
    i = 0
    j = 0
    while i < len(a) and j < len(b):
        if is_word(a[i]) and is_word(b[j]) and a[i].lower() == b[j].lower():
            out[i] = apply_case_pattern(a[i], b[j])
            i += 1
            j += 1
            continue
        if not is_word(a[i]):
            i += 1
            continue
        if not is_word(b[j]):
            j += 1
            continue

        # lookahead
        matched = False
        for k in range(1, 4):
            if i + k < len(a) and is_word(a[i+k]) and a[i+k].lower() == b[j].lower():
                i += k
                matched = True
                break
            if j + k < len(b) and is_word(b[j+k]) and a[i].lower() == b[j+k].lower():
                j += k
                matched = True
                break
        if not matched:
            i += 1
            j += 1

    return join_tokens(out)


# -------------------------
# Prompt builders
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


def build_user_prompt(
    ref: str,
    original: str,
    heritage_kjv: bool,
    kjv_text: Optional[str],
    locks: Optional[List[str]],
) -> str:
    locks = locks or []
    locks_block = ""
    if locks:
        locks_block = (
            "\nHISTORIC PHRASE LOCKS (must appear verbatim if faithful in context):\n"
            + "\n".join([f"- {lk}" for lk in locks])
            + "\n"
        )

    if heritage_kjv and kjv_text:
        return (
            f"REFERENCE: {ref}\n"
            f"Task: Revise the English verse for oral readability, cadence, and recognizability.\n"
            f"Constraints: do NOT change meaning or theology. Do NOT add commentary.\n"
            f"Style: Preserve or recover widely recognized historic phrasing when faithful.\n"
            f"Avoid archaic pronouns (thou/thee/thy). Mild older phrasing is acceptable if remembered.\n"
            f"Output only the revised verse text as ONE line.\n"
            f"{locks_block}\n"
            f"KJV CADENCE ANCHOR (public domain):\n{kjv_text}\n\n"
            f"CURRENT VERSE TO REVISE:\n{original}\n"
        )

    return (
        f"REFERENCE: {ref}\n"
        f"Task: Revise the following English verse for cadence and beauty.\n"
        f"Constraints: do NOT change meaning or theology. Do NOT add commentary.\n"
        f"Output only the revised verse text as ONE line.\n"
        f"{locks_block}\n"
        f"CURRENT VERSE:\n{original}\n"
    )


# -------------------------
# Main
# -------------------------

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

    parser.add_argument("--enforce", action="store_true")
    parser.add_argument("--enforce_only", action="store_true")

    parser.add_argument("--kjv_path", default="")
    parser.add_argument("--heritage_kjv", action="store_true")
    parser.add_argument("--heritage_anchors", default="")

    # Targets-only KJV case normalization
    parser.add_argument("--kjv_case", action="store_true", help="Copy casing from KJV for matched tokens (targets only)")

    args = parser.parse_args()

    targets = load_targets_jsonl(args.targets)
    rows = load_book_jsonl(args.inp)
    charter_text = read_text_file(args.charter)
    system_prompt = build_system_prompt(charter_text)

    kjv_map: Dict[str, str] = {}
    if args.heritage_kjv or args.kjv_case:
        if not args.kjv_path:
            raise ValueError("--heritage_kjv/--kjv_case requires --kjv_path")
        if not os.path.isfile(args.kjv_path):
            raise FileNotFoundError(f"KJV file not found: {args.kjv_path}")
        if os.path.getsize(args.kjv_path) == 0:
            raise ValueError(f"KJV file is empty (0 bytes): {args.kjv_path}")
        kjv_map = load_kjv_jsonl(args.kjv_path)

    anchors: Dict[str, List[str]] = {}
    if args.heritage_anchors:
        anchors = load_heritage_anchors_jsonl(args.heritage_anchors)

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not args.enforce_only and not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing or empty")

    client = OpenAI(api_key=api_key) if not args.enforce_only else None

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    changed = 0
    blocked = 0
    lock_misses = 0
    missing_kjv_for_targets = 0

    with open(args.out, "w", encoding="utf-8") as fout:
        for row in rows:
            ref = row["ref"]
            original = strip_internal_linebreaks(row["translation"])
            out_text = original

            in_targets = ref in targets
            should_model = in_targets and (not args.enforce_only)

            locks = anchors.get(ref, [])
            kjv_text = kjv_map.get(ref) if kjv_map else None
            if in_targets and (args.heritage_kjv or args.kjv_case) and not kjv_text:
                missing_kjv_for_targets += 1

            if should_model:
                user_prompt = build_user_prompt(ref, original, args.heritage_kjv, kjv_text, locks)

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

                # Enforce heritage locks (targets only)
                if locks:
                    revised2, missing_after = enforce_phrase_locks(revised, locks, kjv_text)
                    if missing_after:
                        lock_misses += 1
                    revised = revised2
                    if args.enforce:
                        revised = apply_enforcement(revised)

                # KJV case normalization (targets only)
                if args.kjv_case and kjv_text:
                    revised = kjv_case_normalize_v2(revised, kjv_text)

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
                # Not in targets. Only apply enforcement if enforce_only is set.
                if args.enforce_only and args.enforce:
                    out_text = apply_enforcement(out_text)

                # Heritage locks / KJV case do NOT apply outside targets in this version.

            fout.write(json.dumps({"ref": ref, "translation": out_text}, ensure_ascii=False) + "\n")

            if args.sleep:
                time.sleep(args.sleep)

    print(
        f"Polish complete. Changed: {changed} | Guard-blocked: {blocked} | "
        f"Lock misses: {lock_misses} | Missing KJV refs (targets): {missing_kjv_for_targets}"
    )


if __name__ == "__main__":
    main()
