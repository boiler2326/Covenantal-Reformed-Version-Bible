#!/usr/bin/env python3
"""
Phase 1 translator: OSHB Hebrew source -> English translation JSONL.

Input JSONL lines:
  {"ref":"GEN 1:1","source":"...hebrew..."}
Output JSONL lines:
  {"ref":"GEN 1:1","translation":"In the beginning ..."}

Key behavior:
- Enforces a KJV-inspired but modern-readable register.
- Avoids strong archaisms (thou/thee/ye/hath/doth/etc.) via hard-fail.
- Mild archaisms (e.g., 'unto') are auto-normalized to modern equivalents
  to avoid wasting a whole book run.
- Optional lexical locks can be configured to nudge famous phrases.

NOTE: This script does not attempt divine pronoun capitalization logic.
That is handled later via KJV normalization (your Option 2).
"""

import argparse
import json
import os
import re
import sys
import time
from typing import Dict, List, Optional, Set, Tuple

from openai import OpenAI


# ----------------------------
# Text utilities
# ----------------------------

def normalize_space(s: str) -> str:
    s = re.sub(r"[\r\n]+", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


# ----------------------------
# Mild archaism auto-normalization
# ----------------------------

MILD_ARCHAIC_REPLACEMENTS: List[Tuple[str, str]] = [
    # common mild KJV-ish terms that don't meaningfully change theology
    (r"\bunto\b", "to"),
    (r"\bbetwixt\b", "between"),
    (r"\bnigh\b", "near"),
    (r"\bthereof\b", "of it"),
    (r"\btherein\b", "in it"),
    (r"\btherewith\b", "with it"),
    (r"\bwhence\b", "from where"),
    (r"\bhenceforth\b", "from now on"),
]

def dearchaicize(text: str) -> Tuple[str, List[str]]:
    """
    Apply mild modernizations. Returns (new_text, changed_terms)
    """
    out = text
    changed: List[str] = []
    for pat, rep in MILD_ARCHAIC_REPLACEMENTS:
        if re.search(pat, out, flags=re.IGNORECASE):
            changed.append(pat.strip(r"\b"))
            out = re.sub(pat, rep, out, flags=re.IGNORECASE)
    if out != text:
        out = normalize_space(out)
    return out, changed


# ----------------------------
# Strong archaism hard-block
# ----------------------------

# These are "too-KJV" for your current register goals; hard fail to keep consistency.
STRONG_ARCHAIC_TERMS = [
    "thou", "thee", "thy", "thine",
    "ye", "hath", "doth", "art", "shalt", "wilt",
    "wherefore",
]

STRONG_ARCHAIC_RE = re.compile(
    r"\b(" + "|".join(re.escape(t) for t in STRONG_ARCHAIC_TERMS) + r")\b",
    flags=re.IGNORECASE
)

def find_strong_archaism(text: str) -> Optional[str]:
    m = STRONG_ARCHAIC_RE.search(text)
    return m.group(1).lower() if m else None


# ----------------------------
# Optional lexical locks
# ----------------------------

# Keep this conservative. You can add more as you go.
# Each lock is (substring that must appear, helpful description).
# These are *soft* by default (warn only); set --lock_mode hard to fail.
DEFAULT_SOFT_LOCKS: Dict[str, List[str]] = {
    # Example: Genesis 1:2 historically recognizable phrase
    "GEN 1:2": ["formless and void"],
}

def check_locks(ref: str, text: str, lock_mode: str) -> None:
    locks = DEFAULT_SOFT_LOCKS.get(ref)
    if not locks:
        return
    for lock in locks:
        if lock.lower() not in text.lower():
            msg = f"Soft lock '{lock}' not found in {ref}"
            if lock_mode == "hard":
                raise ValueError(msg)
            else:
                print(f"WARNING: {msg}", file=sys.stderr)


# ----------------------------
# IO helpers
# ----------------------------

def load_input_jsonl(path: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "ref" not in obj or "source" not in obj:
                raise ValueError(f"{path}:{i} missing 'ref' and/or 'source'")
            rows.append({"ref": str(obj["ref"]).strip(), "source": str(obj["source"])})
    return rows


def load_existing_output_refs(path: str) -> Set[str]:
    refs: Set[str] = set()
    if not os.path.exists(path):
        return refs
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            ref = (obj.get("ref") or "").strip()
            if ref:
                refs.add(ref)
    return refs


# ----------------------------
# Prompts
# ----------------------------

SYSTEM_PROMPT = """You are producing a fresh English translation from the original-language source text (Hebrew/Aramaic for OT).
The translation must be:
- Faithful and literal, yet readable to modern readers.
- Reverent and elevated in register (KJV-inspired beauty), but avoid strong archaisms (no thee/thou/ye/hath/doth/etc.).
- No verse numbers, headings, footnotes, commentary, or explanations.
- Output ONLY the translated verse text.

Style:
- Prefer recognizable ecclesiastical phrasing where it is widely known (without copying any modern copyrighted translation).
- Keep divine names consistent: use "the LORD" for YHWH where appropriate in OT context.
- Avoid slang; maintain solemn cadence.

Return only the English verse text, nothing else.
"""

def build_user_prompt(ref: str, source: str) -> str:
    return (
        f"REFERENCE: {ref}\n"
        f"ORIGINAL SOURCE (Hebrew):\n{source}\n\n"
        f"Task: Translate this verse into English.\n"
        f"Constraints: No commentary; output only the verse text.\n"
    )


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 1 translator (OSHB -> English)")
    ap.add_argument("--in", dest="inp", required=True, help="Input jsonl (ref, source)")
    ap.add_argument("--out", dest="out", required=True, help="Output jsonl (ref, translation)")
    ap.add_argument("--model", default="gpt-5.1")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max_output_tokens", type=int, default=280)
    ap.add_argument("--sleep", type=float, default=0.0)
    ap.add_argument("--resume", action="store_true", help="Skip refs already present in --out")
    ap.add_argument("--lock_mode", choices=["off", "soft", "hard"], default="soft")
    args = ap.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing or empty")

    client = OpenAI(api_key=api_key)

    rows = load_input_jsonl(args.inp)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    already_done: Set[str] = set()
    if args.resume:
        already_done = load_existing_output_refs(args.out)

    total = 0
    written = 0
    normalized_count = 0
    strict_failures = 0

    mode = "a" if args.resume and os.path.exists(args.out) else "w"

    with open(args.out, mode, encoding="utf-8") as fout:
        for row in rows:
            ref = row["ref"]
            source = row["source"]
            total += 1

            if args.resume and ref in already_done:
                continue

            user_prompt = build_user_prompt(ref, source)

            resp = client.responses.create(
                model=args.model,
                input=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=args.temperature,
                max_output_tokens=args.max_output_tokens,
            )

            out_text = (resp.output_text or "").strip()
            out_text = normalize_space(out_text)

            # Mild archaism auto-normalization (prevents wasted reruns)
            out_text2, changed_terms = dearchaicize(out_text)
            if out_text2 != out_text:
                normalized_count += 1
                print(f"WARNING: Auto-normalized mild archaism in {ref}: {', '.join(changed_terms)}", file=sys.stderr)
                out_text = out_text2

            # Strong archaism hard block
            bad = find_strong_archaism(out_text)
            if bad is not None:
                strict_failures += 1
                raise ValueError(f"Forbidden archaic term '{bad}' in {ref}")

            # Locks (optional)
            if args.lock_mode != "off":
                check_locks(ref, out_text, args.lock_mode)

            # Write output
            fout.write(json.dumps({"ref": ref, "translation": out_text}, ensure_ascii=False) + "\n")
            written += 1

            if args.sleep:
                time.sleep(args.sleep)

    print(
        f"Phase-1 complete. total_seen={total} written={written} "
        f"mild_normalized={normalized_count} strict_failures={strict_failures}"
    )


if __name__ == "__main__":
    main()
