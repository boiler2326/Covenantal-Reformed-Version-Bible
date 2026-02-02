# scripts/oshb_to_jsonl.py
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

NS = {"osis": "http://www.bibletechnologies.net/2003/OSIS/namespace"}

def normalize_hebrew(s: str) -> str:
    # collapse whitespace; keep cantillation/vowels as-is
    s = re.sub(r"\s+", " ", s).strip()
    return s

def main(xml_path: str, out_path: str):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # OSIS commonly stores verses as <verse osisID="Gen.1.1"> ... </verse>
    verses = root.findall(".//osis:verse", NS)
    if not verses:
        raise RuntimeError("No <verse> elements found. Check OSIS namespace/path.")

    with out_file.open("w", encoding="utf-8") as f:
        for v in verses:
            osis_id = v.get("osisID") or v.get("osisRef") or ""
            if not osis_id:
                continue

            # Extract visible text (ignore tags)
            text = "".join(v.itertext())
            text = normalize_hebrew(text)

            # Convert "Gen.1.1" -> "GEN 1:1" (your format)
            # (Adjust mapping if your refs differ.)
            parts = osis_id.split(".")
            if len(parts) >= 3:
                book, chap, verse = parts[0], parts[1], parts[2]
                ref = f"{book.upper()} {int(chap)}:{int(verse)}"
            else:
                ref = osis_id

            f.write(json.dumps({"ref": ref, "source": text}, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python scripts/oshb_to_jsonl.py sources/oshb/Gen.xml input/Genesis.jsonl")
        sys.exit(2)
    main(sys.argv[1], sys.argv[2])
