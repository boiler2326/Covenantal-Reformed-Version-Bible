#!/usr/bin/env python3
import json
import re
from pathlib import Path

INPUT = "output_phase2/genesis.jsonl"
OUTPUT = "usfm/GEN.usfm"

REF_RE = re.compile(r"^GEN\s+(\d+):(\d+)$")

def main():
    Path("usfm").mkdir(exist_ok=True)

    lines = []
    current_chapter = None

    with open(INPUT, "r", encoding="utf-8") as f:
        for raw in f:
            obj = json.loads(raw)
            ref = obj["ref"]
            text = obj["translation"]

            m = REF_RE.match(ref)
            if not m:
                continue

            chapter = int(m.group(1))
            verse = int(m.group(2))

            if chapter != current_chapter:
                lines.append(f"\\c {chapter}")
                current_chapter = chapter

            lines.append(f"\\v {verse} {text}")

    with open(OUTPUT, "w", encoding="utf-8") as out:
        out.write("\n".join(lines) + "\n")

if __name__ == "__main__":
    main()
