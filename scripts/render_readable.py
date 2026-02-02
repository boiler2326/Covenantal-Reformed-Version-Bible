#!/usr/bin/env python3
import argparse, json, re
from pathlib import Path

REF_RE = re.compile(r"^([A-Z0-9]+)\s+(\d+):(\d+)$")

def parse_ref(ref: str):
    m = REF_RE.match(ref.strip())
    if not m:
        return None
    return m.group(1), int(m.group(2)), int(m.group(3))

def main():
    ap = argparse.ArgumentParser(description="Render JSONL verses to readable text/markdown.")
    ap.add_argument("--in", dest="inp", required=True, help="Input JSONL with {ref, translation}")
    ap.add_argument("--out_txt", required=True, help="Output .txt path")
    ap.add_argument("--out_md", required=True, help="Output .md path")
    ap.add_argument("--title", default="", help="Document title (Markdown)")
    args = ap.parse_args()

    rows = []
    with open(args.inp, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            ref = obj.get("ref","").strip()
            txt = obj.get("translation","").strip()
            k = parse_ref(ref)
            if not k or not txt:
                continue
            book, ch, vs = k
            rows.append((book, ch, vs, txt))

    # Sort defensively
    rows.sort(key=lambda x: (x[1], x[2]))

    out_txt = []
    out_md = []

    if args.title:
        out_md.append(f"# {args.title}\n")

    cur_ch = None
    for book, ch, vs, txt in rows:
        if cur_ch != ch:
            cur_ch = ch
            out_txt.append(f"\nCHAPTER {ch}\n")
            out_md.append(f"\n## Chapter {ch}\n")
        # Verse formatting
        out_txt.append(f"{vs} {txt}")
        out_md.append(f"**{vs}** {txt}")

    Path(args.out_txt).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)

    Path(args.out_txt).write_text("\n".join(out_txt).strip() + "\n", encoding="utf-8")
    Path(args.out_md).write_text("\n".join(out_md).strip() + "\n", encoding="utf-8")

if __name__ == "__main__":
    main()
