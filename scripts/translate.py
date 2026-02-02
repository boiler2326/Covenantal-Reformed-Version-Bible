#!/usr/bin/env python3

import argparse
import json
import os
import re
import time
from openai import OpenAI

HARD_LOCKS = {
    "PHI 3:8": "excrement",
    "REV 3:16": "spew",
    "GEN 3:15": "Seed"
}

SOFT_LOCKS = {
    "GEN 1:1": "In the beginning",
    "GEN 1:3": "Let there be light",
    "GEN 1:2": "without form"
}
FORBIDDEN_ARCHAIC_WORDS = [
    "thee", "thou", "thy", "thine", "ye",
    "hath", "doth", "saith", "unto"
]

FORBIDDEN_ARCHAIC_PAT = re.compile(
    r"\b(" + "|".join(FORBIDDEN_ARCHAIC_WORDS) + r")\b",
    re.IGNORECASE
)

def contains_forbidden_archaic(text):
    m = FORBIDDEN_ARCHAIC_PAT.search(text)
    return m.group(1).lower() if m else None

def check_hard_locks(ref, text):
    if ref in HARD_LOCKS:
        required = HARD_LOCKS[ref]
        if required not in text:
            return required
    return None

def soft_lock_warning(ref, text):
    if ref in SOFT_LOCKS and SOFT_LOCKS[ref] not in text:
        print(f"WARNING: Soft lock '{SOFT_LOCKS[ref]}' not found in {ref}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="inp", required=True)
    parser.add_argument("--out", dest="out", required=True)
    parser.add_argument("--system", default="charter/system_prompt.txt")
    parser.add_argument("--model", default="gpt-5.1")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--sleep", type=float, default=0.0)
    args = parser.parse_args()

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    with open(args.system, "r", encoding="utf-8") as f:
        system_prompt = f.read()

    with open(args.inp, "r", encoding="utf-8") as fin, \
         open(args.out, "w", encoding="utf-8") as fout:

        for line in fin:
            verse = json.loads(line)

            user_prompt = f"""
REFERENCE: {verse['ref']}
Translate the following SOURCE TEXT into Modern Sacral English under the established charter.
Output only the translated verse text. Do NOT include the verse number.

SOURCE TEXT:
{verse['source']}
""".strip()

            response = client.responses.create(
                model=args.model,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=args.temperature,
                max_output_tokens=400
            )

            text = response.output_text.strip()

            bad = contains_forbidden_archaic(text)
            if bad is not None:
               print(f"Offending output for {verse['ref']}: {text}")
                raise ValueError(f"Forbidden archaic term '{bad}' in {verse['ref']}")

            lock = check_hard_locks(verse["ref"], text)
            if lock:
                raise ValueError(f"Missing HARD lock '{lock}' in {verse['ref']}")

            fout.write(json.dumps({
                "ref": verse["ref"],
                "translation": text
            }, ensure_ascii=False) + "\n")

            soft_lock_warning(verse["ref"], text)

            if args.sleep:
                time.sleep(args.sleep)

if __name__ == "__main__":
    main()
