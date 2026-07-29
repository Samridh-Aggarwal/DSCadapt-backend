"""Read the corpus and turn it into chunks.

Which file is what, and which measure it belongs to, comes from
data/documents.json. app.py decided that four different ways — a hardcoded
filename list, a filename-to-measure dict, a substring check for "Barcelona",
and two module constants — so adding a case study meant editing code.

This module reports problems rather than repairing them. The documents are due
a cleaning pass, and a repair heuristic written against today's mess would
become dead weight, or worse, misfire on the cleaned versions. Every complaint
below is something the cleaning pass should make go away; when it does, the
warning simply stops appearing and no code changes.

Two things here are design, not workarounds, and stay whichever way the
cleaning goes:
  - text before the first numbered section is indexed as an overview chunk.
    Every document has framing prose; the chunker previously had nowhere to
    put it, so it was silently dropped.
  - a duplicate section number never causes content to be discarded. 1.6
    numbers two of its sections twice, and picking the longer one deleted
    both of its trade-offs.
"""

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

# What belongs in the retrieval pool. The scoping review is matched separately
# and injected under its own heading, so indexing it as well would put the same
# document in one context twice, the second time labelled as pathway evidence.
INDEXED_KINDS = {"pathway", "case_study", "glossary", "overview"}

MIN_CHUNK = 200          # below this a section is a stub, not content
MAX_CHUNK = 2000         # sub-split above this
CONTENTS_RUN = 3         # consecutive header-only lines that make a contents list
CONTENTS_WITHIN = 15     # ...and it has to start near the top

# One pattern, used for both splitting and labelling. app.py had two that
# disagreed: the splitter handled three levels, the label parser four, so
# 4.1.1.1 through 4.1.1.7 all collapsed into a chunk labelled 4.1.1.
_LEVELS = r'\d+\.\d+\.\d+(?:\.\d+)*'
HEADER = re.compile(rf'^({_LEVELS})\.?\s+(.*)$')
SPLIT = re.compile(rf'(?=\n{_LEVELS}\.?\s)')

REF_DELIMITERS = [
    (re.compile(r'\n\s*_{5,}\s*\n'), "underscore rule"),
    (re.compile(r'(?im)^[ \t]*BIBLIOGRAPHY:?[ \t]*$'), "BIBLIOGRAPHY heading"),
    (re.compile(r'(?im)^[ \t]*REFERENCES?:?[ \t]*$'), "References heading"),
    (re.compile(r'(?im)^[ \t]*SOURCES?:?[ \t]*$'), "Sources heading"),
    (re.compile(r'(?im)^[ \t]*(?:SUPPORTIVE[ \t]+)?LITERATURE[^\n]{0,40}$'), "Literature heading"),
    (re.compile(r'(?im)^[ \t]*(?:FOOTNOTES?|ENDNOTES?):?[ \t]*$'), "Notes heading"),
]

EDITORIAL = re.compile(r'^\s*(?:N\.?B\.?[ .]|IMP[ :]|TBD\b|Could be |This is not on the diagram)')


@dataclass
class Chunk:
    text: str
    source: str
    section: str
    kind: str = "pathway"


@dataclass
class Corpus:
    chunks: list = field(default_factory=list)
    references: dict = field(default_factory=dict)
    publications: list = field(default_factory=list)
    measures: dict = field(default_factory=dict)   # source filename -> measure
    problems: list = field(default_factory=list)

    def indexed(self):
        """Chunks that belong in the retrieval index."""
        return [c for c in self.chunks if c.kind in INDEXED_KINDS]

    def of_kind(self, kind):
        return [c for c in self.chunks if c.kind == kind]

    def note(self, source, message):
        self.problems.append((source, message))

    def report(self):
        """Printed at startup. An empty report means the corpus is clean."""
        if not self.problems:
            return f"corpus: {len(self.chunks)} chunks, no problems"
        lines = [f"corpus: {len(self.chunks)} chunks, {len(self.problems)} problems"]
        current = None
        for source, message in self.problems:
            if source != current:
                lines.append(f"  {source}")
                current = source
            lines.append(f"    {message}")
        return "\n".join(lines)


# --- shared helpers ----------------------------------------------------------

# Word writes U+2028 for a soft return (shift+enter) and U+2029 for a paragraph
# break. Converters preserve them verbatim. Python's split("\n") does not treat
# either as a line break, so a contents list written with soft returns arrives
# as one physical line and every header in it becomes invisible.
LINE_SEPARATORS = str.maketrans({"\u2028": "\n", "\u2029": "\n\n", "\r": "\n"})


def clean(text):
    text = text.replace("\ufeff", "")
    text = text.translate(LINE_SEPARATORS)
    text = re.sub(r"\[[a-z]\]", "", text)      # Word comment anchors
    return text.strip()


def split_references(text):
    """Return (body, references, which delimiter matched).

    app.py only knew the underscore rule, so Barcelona — which uses a
    BIBLIOGRAPHY heading — contributed no references at all.
    """
    best = None
    for pattern, label in REF_DELIMITERS:
        m = pattern.search(text)
        if m and (best is None or m.start() < best[0].start()):
            best = (m, label)
    if best is None:
        return text.strip(), "", None
    m, label = best
    return text[:m.start()].strip(), text[m.end():].strip(), label


def sub_chunk(text, header, max_chars=MAX_CHUNK):
    """Split a long section on paragraph breaks, restamping the header."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if len(paras) == 1 and len(text) > max_chars:
        paras = [p.strip() for p in text.split("\n") if p.strip()]
    if len(paras) == 1 and len(text) > max_chars:
        paras = [p.strip() for p in re.split(r"(?<=[.!?])\s+", text) if p.strip()]

    out, current = [], header + "\n\n"
    for para in paras:
        if len(current) + len(para) > max_chars and len(current) > len(header) + 10:
            out.append(current.strip())
            current = header + "\n\n"
        current += para + "\n\n"
    if len(current.strip()) > len(header) + 10:
        out.append(current.strip())
    return out


def strip_editorial(lines):
    """Drop authors' notes to each other from indexed text.

    "NB. Potentially best placed in another pathway" is a note to a colleague,
    not evidence. These used to vanish by accident, riding along with the
    discarded preamble; now that the preamble is indexed they would reach the
    model, so they come out deliberately and each removal is reported.
    """
    kept, removed = [], []
    for line in lines:
        if EDITORIAL.match(line):
            removed.append(line.strip()[:70])
        else:
            kept.append(line)
    return kept, removed


def find_contents_block(lines):
    """(start, end) line span of the contents list, or None.

    A contents list is a run of consecutive lines that are nothing but section
    headers. In the body a header is followed by prose, so runs there are of
    length one. Only the run itself is removed — the document title sits above
    it and is what the overview chunk gets named after.

    Left in place, the last contents entry swallows the document's opening
    paragraph and then loses a dedup against the real section with the same
    number, which is how every document's framing prose disappeared.
    """
    run_start = run_len = seen = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            if run_len >= CONTENTS_RUN:
                return run_start, run_start + run_len
            run_len = 0
            continue
        seen += 1
        if HEADER.match(stripped):
            if run_len == 0:
                run_start = i
            run_len += 1
        else:
            if run_len >= CONTENTS_RUN:
                return run_start, run_start + run_len
            run_len = 0
            if seen > CONTENTS_WITHIN:
                break
    return (run_start, run_start + run_len) if run_len >= CONTENTS_RUN else None


# --- parsers -----------------------------------------------------------------

def parse_pathway(source, text, corpus):
    text = clean(text)
    body, refs, delimiter = split_references(text)
    if delimiter is None:
        corpus.note(source, "no reference separator found, citations stay inline")
    corpus.references[source] = refs

    lines = body.split("\n")
    span = find_contents_block(lines)
    if span:
        start, end = span
        corpus.note(source, f"contents list of {end - start} entries stripped before chunking")
        lines = lines[:start] + lines[end:]

    lines, removed = strip_editorial(lines)
    for note in removed:
        corpus.note(source, f"editorial note removed: {note!r}")
    body = "\n".join(lines)

    parts = SPLIT.split("\n" + body)
    preamble, sections = parts[0].strip(), []
    for part in parts[1:]:
        part = part.strip()
        head, _, rest = part.partition("\n")
        m = HEADER.match(head)
        if not m:
            continue
        sections.append((m.group(1).rstrip("."), m.group(2).strip(), part))

    if len(preamble) >= MIN_CHUNK:
        title = preamble.split("\n")[0].strip()
        label = f"{title} — overview"
        for sc in sub_chunk(preamble, label):
            corpus.chunks.append(Chunk(sc, source, label, "overview"))
    elif preamble:
        corpus.note(source, f"preamble too short to index ({len(preamble)} chars)")

    # Keep every distinct section. Only a genuine repeat of the same number AND
    # title is a duplicate; the same number with different titles is a numbering
    # error in the source, and both halves are real content.
    numbers = Counter(sid for sid, _, _ in sections)
    for sid, count in numbers.items():
        if count > 1:
            titles = [t for s, t, _ in sections if s == sid]
            if len(set(titles)) > 1:
                corpus.note(source, f"section {sid} declared {count} times with different "
                                    f"titles {titles}; all kept, source numbering needs fixing")

    longest = {}
    for sid, title, part in sections:
        key = (sid, title)
        if key not in longest or len(part) > len(longest[key]):
            longest[key] = part

    numbered = {sid for sid, _, _ in sections}
    for (sid, title), part in longest.items():
        label = f"{sid} {title}".strip()
        content, _, _ = split_references(part)
        if len(content) > MAX_CHUNK:
            for sc in sub_chunk(content, label):
                corpus.chunks.append(Chunk(sc, source, label, "pathway"))
        elif len(content) >= MIN_CHUNK:
            corpus.chunks.append(Chunk(content, source, label, "pathway"))
        elif any(child.startswith(sid + ".") for child in numbered):
            # A heading whose body lives in its own numbered children. Nothing
            # is lost — 2.1.3 is empty because 2.1.3.1 and 2.1.3.2 hold the text.
            pass
        else:
            corpus.note(source, f"section {label!r} dropped, only {len(content)} chars")


def parse_case_study(source, text, corpus):
    text = clean(text)
    body, refs, delimiter = split_references(text)
    corpus.references[source] = refs
    if delimiter is None:
        corpus.note(source, "no reference separator found, citations stay inline")
    title = body.split("\n")[0].strip() or Path(source).stem
    for sc in sub_chunk(body, title):
        corpus.chunks.append(Chunk(sc, source, title, "case_study"))


def parse_glossary(source, text, corpus):
    """One chunk per term.

    The file holds a tab-indented term/definition table and, below it, numbered
    full definitions. Merged so each term is a single chunk labelled with its
    own name. app.py ran this through the pathway parser, which found no
    numbered headers and produced one 39,000-character section labelled
    'Climate-Sensitive Infectious Diseases in' — truncated mid-word, and
    identical on every chunk.
    """
    text = clean(text)
    lines = text.split("\n")

    short, group, pending = {}, None, None
    full, current, buffer = {}, None, []
    in_full = False

    for line in lines:
        if re.match(r"^Section \d+", line.strip()):
            in_full = "full definition" in line.lower()
            continue

        if not in_full:
            if line.startswith("\t"):
                value = line[1:].strip()
                if value in ("Definition", "Term") or not value:
                    continue
                if pending is None:
                    pending = value
                else:
                    short[pending] = (value, group)
                    pending = None
            elif line.strip() and line.strip() not in ("Term",):
                group = line.strip()
                pending = None
            continue

        m = re.match(r"^\d+\.\s+(\S.*)$", line.strip())
        if m:
            if current:
                full[current] = "\n".join(buffer).strip()
            current, buffer = m.group(1).strip(), []
        elif current is not None:
            buffer.append(line)
    if current:
        full[current] = "\n".join(buffer).strip()

    for term in list(short) + [t for t in full if t not in short]:
        definition, term_group = short.get(term, ("", None))
        body = full.get(term, "")
        parts = [p for p in (definition, body) if p]
        if not parts:
            continue
        label = term if not term_group else f"{term} ({term_group})"
        corpus.chunks.append(Chunk(f"{term}\n\n" + "\n\n".join(parts),
                                   source, label, "glossary"))

    corpus.note(source, f"{len(short)} glossary terms, {len(full)} full definitions, "
                        f"{len(set(short) | set(full))} chunks")


def parse_scoping_review(source, text, corpus, max_chars=1800):
    header_re = re.compile(r"^(\d+(?:\.\d+)*)\s{2,}(\S.*)$")
    name, buf, raw = None, [], []
    for line in text.split("\n"):
        s = line.rstrip()
        m = header_re.match(s)
        if m and not re.search(r"\.{4,}", s):          # dot leaders mean contents page
            title = m.group(2).strip()
            if title.lower().startswith(("reference", "appendix", "appendices")):
                break
            if name:
                raw.append((name, " ".join(buf).strip()))
            name, buf = f"D2.2 §{m.group(1)} {title}", []
            continue
        if name is not None:
            t = s.strip()
            if t and set(t) != {"_"}:
                buf.append(t)
    if name:
        raw.append((name, " ".join(buf).strip()))

    kept = 0
    for section, body in raw:
        if len(body) < MIN_CHUNK:
            continue
        pieces = [body] if len(body) <= max_chars else _split_sentences(body, max_chars)
        for piece in pieces:
            corpus.chunks.append(Chunk(piece, source, section, "scoping_review"))
        kept += 1
    corpus.note(source, f"{kept} of {len(raw)} sections indexed")


def _split_sentences(text, max_chars):
    out, piece = [], ""
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        if piece and len(piece) + len(sentence) + 1 > max_chars:
            out.append(piece.strip())
            piece = sentence
        else:
            piece = (piece + " " + sentence).strip()
    if piece.strip():
        out.append(piece.strip())
    return out


FIELDS = {"Authors:": "authors", "Citation:": "citation", "Date:": "date",
          "Type:": "type", "Link:": "link"}
BODY_FIELDS = ("Abstract:", "Deliverable Summary:", "Summary:", "Full text:", "Key messages:")


def parse_publications(source, text, corpus):
    """Project outputs. Kept out of the chunk pool and matched separately."""
    for block in (b.strip() for b in text.split("---") if b.strip()):
        lines = [l.strip() for l in block.split("\n") if l.strip()]
        if not lines or lines[0].startswith(("1. Deliverables", "2. Publications", "Source:")):
            continue

        entry = {"title": re.sub(r"^\d+[.)]\s*", "", lines[0]).strip()}
        body, reading = [], False
        for line in lines[1:]:
            matched = next((f for f in FIELDS if line.startswith(f)), None)
            if matched:
                entry[FIELDS[matched]] = line[len(matched):].strip()
                reading = False
            elif line.startswith(BODY_FIELDS):
                head = line.split(":", 1)[1].strip()
                if head:
                    body.append(head)
                reading = True
            elif reading:
                body.append(line)

        entry["text"] = " ".join(body) if body else entry["title"]
        entry["short"] = _short_citation(entry)
        if not entry.get("link"):
            corpus.note(source, f"no link, dropped: {entry['title'][:60]!r}")
            continue
        if not body:
            corpus.note(source, f"no summary, indexed on title alone: {entry['title'][:60]!r}")
        corpus.publications.append(entry)

    corpus.note(source, f"{len(corpus.publications)} publications indexed")


def _short_citation(entry):
    year = (entry.get("date") or "")[-4:]
    authors = entry.get("authors")
    if authors and year:
        # surname only; the original used the full first name
        first = authors.split(",")[0].strip()
        surname = first.split()[-1] if first.split() else first
        many = "et al" in authors or authors.count(",") > 1
        return f"{surname} et al. ({year})" if many else f"{surname} ({year})"
    if year:
        title = entry["title"]
        ident = title.split("\u2014")[0].strip() if "\u2014" in title else "Deliverable"
        return f"IDAlert {ident} ({year})"
    return entry["title"][:50]


PARSERS = {
    "pathway": parse_pathway,
    "case_study": parse_case_study,
    "glossary": parse_glossary,
    "scoping_review": parse_scoping_review,
    "publications": parse_publications,
}


# --- entry point -------------------------------------------------------------

def load(root=None, manifest=None, known_measures=None):
    """Read every document the manifest lists. Missing files are reported, not fatal."""
    root = Path(root or Path(__file__).parent / "docs")
    manifest = Path(manifest or Path(__file__).parent / "data" / "documents.json")
    entries = json.loads(manifest.read_text(encoding="utf-8"))["documents"]

    corpus = Corpus()
    listed = set()

    for entry in entries:
        rel, kind = entry["file"], entry["type"]
        name = Path(rel).name
        listed.add(rel)

        if kind not in PARSERS:
            corpus.note(name, f"unknown type {kind!r}")
            continue
        if entry.get("measure"):
            corpus.measures[name] = entry["measure"]
            if known_measures is not None and entry["measure"] not in known_measures:
                corpus.note(name, f"measure {entry['measure']!r} is not in the ontology")

        path = root / rel
        if not path.exists():
            corpus.note(name, "listed in the manifest but not on disk")
            continue
        PARSERS[kind](name, clean(path.read_text(encoding="utf-8-sig")), corpus)

    for path in sorted(root.rglob("*.txt")):
        rel = str(path.relative_to(root))
        if rel not in listed:
            corpus.note(rel, "on disk but not in the manifest, so never indexed")

    return corpus
