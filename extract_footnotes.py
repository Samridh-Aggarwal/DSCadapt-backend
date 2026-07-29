"""Pull footnotes and endnotes out of .docx files and append them as references.

Word keeps footnotes in a separate part of the file. textutil, and most other
converters, read only the main document body and drop them — which for these
pathway documents means dropping every citation, because that is where the
citations live.

This reads word/footnotes.xml and word/endnotes.xml straight out of the .docx
(they are zip archives) and appends what it finds to the matching .txt under a
References heading, which is what the corpus loader looks for.

Run it from the dscadapt-backend folder:

    python3 extract_footnotes.py

It only ever appends, and it skips any file that already has a References
heading, so running it twice is safe.
"""

import re
import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
FOLDERS = ["backend/docs/pathways", "backend/docs/case_studies", "backend/docs/reference"]


def notes_from(docx, part):
    """Text of every real note in one part. Skips Word's separator entries."""
    try:
        with zipfile.ZipFile(docx) as archive:
            xml = archive.read(f"word/{part}.xml")
    except (KeyError, zipfile.BadZipFile):
        return []

    root = ElementTree.fromstring(xml)
    out = []
    for note in root:
        if note.get(f"{W}type") in ("separator", "continuationSeparator"):
            continue
        text = "".join(t.text or "" for t in note.iter(f"{W}t")).strip()
        text = re.sub(r"\s+", " ", text)
        if len(text) > 10:
            out.append(text)
    return out


def main():
    total_files = total_notes = 0
    for folder in FOLDERS:
        for docx in sorted(Path(folder).glob("*.docx")):
            target = docx.with_suffix(".docx.txt")
            if not target.exists():
                print(f"  skip, no .txt yet: {docx.name}")
                continue

            body = target.read_text(encoding="utf-8")
            if re.search(r"(?im)^\s*references\s*$", body):
                continue

            notes = notes_from(docx, "footnotes") + notes_from(docx, "endnotes")
            if not notes:
                print(f"  no notes found: {docx.name}")
                continue

            numbered = "\n".join(f"{i}. {n}" for i, n in enumerate(notes, 1))
            target.write_text(body.rstrip() + "\n\nReferences\n" + numbered + "\n",
                              encoding="utf-8")
            print(f"  {len(notes):>3} references -> {docx.name}")
            total_files += 1
            total_notes += len(notes)

    print(f"\n{total_notes} references appended across {total_files} documents")


if __name__ == "__main__":
    main()
