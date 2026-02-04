#!/usr/bin/env python3
"""
Build sources/kjv/kjv.jsonl from an OSIS KJV file.

Output JSONL lines look like:
{"ref":"PSA 23:1","kjv":"The LORD is my shepherd; I shall not want."}

This parser handles:
- <verse osisID="Ps.23.1"> ... </verse>
- milestone verses using sID/eID (start/end)
"""

import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple

# Map OSIS book IDs -> your 3-letter refs
OSIS_TO_REF = {
    # OT
    "Gen": "GEN",
    "Exod": "EXO",
    "Lev": "LEV",
    "Num": "NUM",
    "Deut": "DEU",
    "Josh": "JOS",
    "Judg": "JDG",
    "Ruth": "RUT",
    "1Sam": "1SA",
    "2Sam": "2SA",
    "1Kgs": "1KI",
    "2Kgs": "2KI",
    "1Chr": "1CH",
    "2Chr": "2CH",
    "Ezra": "EZR",
    "Neh": "NEH",
    "Esth": "EST",
    "Job": "JOB",
    "Ps": "PSA",
    "Prov": "PRO",
    "Eccl": "ECC",
    "Song": "SNG",
    "Isa": "ISA",
    "Jer": "JER",
    "Lam": "LAM",
    "Ezek": "EZK",
    "Dan": "DAN",
    "Hos": "HOS",
    "Joel": "JOL",
    "Amos": "AMO",
    "Obad": "OBA",
    "Jonah": "JON",
    "Mic": "MIC",
    "Nah": "NAM",
    "Hab": "HAB",
    "Zeph": "ZEP",
    "Hag": "HAG",
    "Zech": "ZEC",
    "Mal": "MAL",
    # NT
    "Matt": "MAT",
    "Mark": "MRK",
    "Luke": "LUK",
    "John": "JHN",
    "Acts": "ACT",
    "Rom": "ROM",
    "1Cor": "1CO",
    "2Cor": "2CO",
    "Gal": "GAL",
    "Eph": "EPH",
    "Phil": "PHP",
    "Col": "COL",
    "1Thess": "1TH",
    "2Thess": "2TH",
    "1Tim": "1TI",
    "2Tim": "2TI",
    "Titus": "TIT",
    "Phlm": "PHM",
    "Heb": "HEB",
    "Jas": "JAS",
    "1Pet": "1PE",
    "2Pet": "2PE",
    "1John": "1JN",
    "2John": "2JN",
    "3John": "3JN",
    "Jude": "JUD",
    "Rev": "REV",
}

WHITESPACE_RE = re.compile(r"\s+")
BRACKETED_RE = re.compile(r"\[[^\]]*\]")  # sometimes OSIS has bracketed notes
STRIP_PUNCT_SPACE_RE = re.compile(r"\s+([,.;:!?])")
FIX_QUOTES_RE = re.compile(r"\s+([’”])")  # tighten spacing before closing quotes

def normalize_text(s: str) -> str:
    s = s.replace("\u00a0", " ")
    s = WHITESPACE_RE.sub(" ", s).strip()
    s = BRACKETED_RE.sub("", s).strip()
    s = STRIP_PUNCT_SPACE_RE.sub(r"\1", s)
    s = FIX_QUOTES_RE.sub(r"\1", s)
    return s.strip()

def osis_id_to_ref(osis_id: str) -> Optional[str]:
    """
    OSIS verse id usually like: Ps.23.1 or 1Sam.17.4 etc
    """
    parts = osis_id.split(".")
    if len(parts) < 3:
        return None
    book = parts[0]
    chap = parts[1]
    verse = parts[2]
    ref_book = OSIS_TO_REF.get(book)
    if not ref_book:
        return None
    # normalize numeric
    try:
        c = int(chap)
        v = int(verse)
    except ValueError:
        return None
    return f"{ref_book} {c}:{v}"

def iter_text_excluding(elem: ET.Element, exclude_tags: Tuple[str, ...] = ("note",)) -> str:
    """
    Collect element text but skip excluded subtrees (notes).
    """
    chunks: List[str] = []

    def walk(e: ET.Element):
        tag = e.tag.split("}")[-1]  # strip namespace
        if tag in exclude_tags:
            return
        if e.text:
            chunks.append(e.text)
        for child in list(e):
            walk(child)
            if child.tail:
                chunks.append(child.tail)

    walk(elem)
    return "".join(chunks)

def parse_osis(osis_path: str) -> Dict[str, str]:
    tree = ET.parse(osis_path)
    root = tree.getroot()

    # Find all <verse> elements regardless of namespace
    verses = root.findall(".//{*}verse")

    out: Dict[str, str] = {}

    # Handle milestone verses: <verse sID="X"/> ... <verse eID="X"/>
    active_id: Optional[str] = None
    active_buf: List[str] = []
    active_ref: Optional[str] = None

    def flush_active():
        nonlocal active_id, active_buf, active_ref
        if active_ref and active_buf:
            txt = normalize_text("".join(active_buf))
            if txt:
                out[active_ref] = txt
        active_id = None
        active_ref = None
        active_buf = []

    # Build a linear traversal by walking all elements in document order
    # We’ll use root.iter() and detect verse milestones.
    for e in root.iter():
        tag = e.tag.split("}")[-1]
        if tag == "verse":
            osisID = e.attrib.get("osisID")
            sID = e.attrib.get("sID")
            eID = e.attrib.get("eID")

            if osisID and not (sID or eID):
                # normal container verse
                ref = osis_id_to_ref(osisID)
                if ref:
                    txt = normalize_text(iter_text_excluding(e))
                    if txt:
                        out[ref] = txt
                continue

            if sID:
                # start milestone
                flush_active()
                active_id = sID
                active_ref = osis_id_to_ref(sID)
                continue

            if eID:
                # end milestone
                if active_id and eID == active_id:
                    flush_active()
                continue

        # If we’re inside a milestone verse, capture text from non-verse nodes
        if active_id:
            # Skip notes entirely
            if tag == "note":
                continue
            if e.text:
                active_buf.append(e.text)
            # tails are handled naturally by iter, but include tail too:
            if e.tail:
                active_buf.append(e.tail)

    # flush in case file ends unexpectedly
    flush_active()

    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("osis_xml", help="Path to OSIS XML (KJV)")
    ap.add_argument("out_jsonl", help="Output JSONL path")
    args = ap.parse_args()

    mapping = parse_osis(args.osis_xml)
    if not mapping:
        print("ERROR: produced 0 verses; OSIS parsing failed or mappings missing.", file=sys.stderr)
        sys.exit(2)

    # Deterministic order: sort by book then chapter/verse
    def sort_key(ref: str):
        book, cv = ref.split(" ", 1)
        c, v = cv.split(":")
        return (book, int(c), int(v))

    refs = sorted(mapping.keys(), key=sort_key)

    with open(args.out_jsonl, "w", encoding="utf-8") as f:
        for ref in refs:
            f.write(json.dumps({"ref": ref, "kjv": mapping[ref]}, ensure_ascii=False) + "\n")

    print(f"Wrote {len(refs)} verses to {args.out_jsonl}")

if __name__ == "__main__":
    main()
