#!/usr/bin/env python3

import argparse
import json
import os
import re
import time
from openai import OpenAI

FORBIDDEN_ARCHAIC = [
    "thee", "thou", "thy", "thine", "ye",
    "hath", "doth", "saith"
]

LEXICAL_LOCKS = {
    "GEN 1:2": "without form",
    "GEN 1:1": "In the beginning",
    "GEN 1:3": "Let there be light",
    "PHI 3:8": "excrement",
    "REV 3:16": "spew",
    "GEN 3:15": "Seed"
}
def contains_forbidden_archaic(text):
    lower = text.lower()
    for word in FORBIDDEN_ARCHAIC:
        if word in lower:
            return word
    return None

def check_lexical_locks(ref, text):
    if ref in LEXICAL_LOCKS:
        required = LEXICAL_LOCKS[ref]
        if required not in text:
            return required
    return None

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
            if bad:
                raise ValueError(f"Forbidden archaic term '{bad}' in {verse['ref']}")

            lock = check_lexical_locks(verse["ref"], text)
            if lock:
                raise ValueError(f"Missing lexical lock '{lock}' in {verse['ref']}")

            fout.write(json.dumps({
                "ref": verse["ref"],
                "translation": text
            }, ensure_ascii=False) + "\n")

            if args.sleep:
                time.sleep(args.sleep)

if __name__ == "__main__":
    main()
