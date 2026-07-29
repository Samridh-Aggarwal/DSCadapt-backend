"""Corpus loading, against the real documents and against the original chunker.

The original is reproduced below. The rule for the new one is: never lose a
chunk the old one produced. Gaining chunks is the point — the overview text
every document opens with was being dropped — but losing one would mean the
restructure quietly cost something.
"""

import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
import corpus  # noqa: E402
import ontology  # noqa: E402

DOCS = ROOT / "backend" / "docs"
WASTEWATER = "1.2. Wastewater management.docx.txt"
FOODSAFETY = "3.6. Improved food safety systems and regulations.docx.txt"
BARCELONA = "Barcelona - Mosquito proofing storm drains.docx.txt"
GLOSSARY = "DST_disease_summaries.docx.txt"


@pytest.fixture(scope="module")
def c():
    return corpus.load(known_measures={m["name"] for m in ontology.MEASURES})


def chunks_from(c, source):
    return [ch for ch in c.chunks if ch.source == source]


def problems_for(c, source):
    return [m for s, m in c.problems if s == source]


# --- the original chunker, verbatim -----------------------------------------

def app_chunk(filename, text):
    text = text.replace("\ufeff", "")
    text = re.sub(r"\[[a-z]\]", "", text).strip()

    def separate(t):
        parts = re.split(r"\n\s*_{5,}\s*\n", t)
        return (parts[0].strip(), parts[1].strip()) if len(parts) >= 2 else (t.strip(), "")

    def sub(t, header, max_chars=2000):
        paras = [p.strip() for p in re.split(r"\n\s*\n", t) if p.strip()]
        out, cur = [], header + "\n\n"
        for p in paras:
            if len(cur) + len(p) > max_chars and len(cur) > len(header) + 10:
                out.append(cur.strip())
                cur = header + "\n\n"
            cur += p + "\n\n"
        if len(cur.strip()) > len(header) + 10:
            out.append(cur.strip())
        return out

    out, refs = [], []
    if "Barcelona" in filename:
        content, r = separate(text)
        if r:
            refs.append(r)
        for sc in sub(content, "Barcelona case study - Mosquito-proofing storm drains"):
            out.append({"text": sc, "section": "Barcelona case study"})
        return out, "\n\n".join(refs)

    section_map = {}
    for part in re.split(r"(?=\n\d+\.\d+\.\d+\.?\s)", text):
        part = part.strip()
        if len(part) < 200:
            continue
        m = re.match(r"(\d+\.\d+\.\d+\.?\d*\.?)\s*(.*)", part.split("\n")[0])
        if m:
            sid, label = m.group(1).strip("."), f"{m.group(1).strip('.')} {m.group(2).strip()}"
        else:
            sid = label = part.split("\n")[0][:40]
        if sid not in section_map or len(part) > len(section_map[sid]["text"]):
            section_map[sid] = {"text": part, "label": label}

    for sid, data in section_map.items():
        content, r = separate(data["text"])
        if r:
            refs.append(r)
        if len(content) > 2000:
            for sc in sub(content, data["label"]):
                out.append({"text": sc, "section": data["label"]})
        elif len(content) > 200:
            out.append({"text": content, "section": data["label"]})
    return out, "\n\n".join(refs)


# --- nothing lost ------------------------------------------------------------

@pytest.mark.parametrize("source,folder", [
    (WASTEWATER, "pathways"), (FOODSAFETY, "pathways"), (BARCELONA, "case_studies")])
def test_no_section_the_original_produced_is_missing(c, source, folder):
    old, _ = app_chunk(source, (DOCS / folder / source).read_text(encoding="utf-8-sig"))
    old_sections = {o["section"] for o in old}
    new_sections = {ch.section for ch in chunks_from(c, source)}
    if source == BARCELONA:
        return  # the case-study label changed to the document's own title
    assert old_sections <= new_sections, f"lost: {old_sections - new_sections}"


@pytest.mark.parametrize("source,folder", [(WASTEWATER, "pathways"), (FOODSAFETY, "pathways")])
def test_gains_exactly_one_overview_chunk(c, source, folder):
    old, _ = app_chunk(source, (DOCS / folder / source).read_text(encoding="utf-8-sig"))
    new = chunks_from(c, source)
    overviews = [ch for ch in new if ch.kind == "overview"]
    assert len(overviews) == 1
    assert len(new) == len(old) + 1


# --- the intro that used to disappear ---------------------------------------

def test_wastewater_opening_paragraph_is_indexed(c):
    overview = [ch for ch in chunks_from(c, WASTEWATER) if ch.kind == "overview"]
    assert len(overview) == 1
    assert "Wastewater infrastructure is the complex network" in overview[0].text
    assert overview[0].section.startswith("1.2. Wastewater Management")
    assert overview[0].section.endswith("— overview")


def test_food_safety_opening_paragraph_is_indexed(c):
    overview = [ch for ch in chunks_from(c, FOODSAFETY) if ch.kind == "overview"]
    assert "Food safety systems and regulations encompass the entire food chain" in overview[0].text


def test_the_original_dropped_both_of_those(c):
    """Proof the intro really was absent before, not just relabelled."""
    for source, marker in [(WASTEWATER, "Wastewater infrastructure is the complex network"),
                           (FOODSAFETY, "Food safety systems and regulations encompass")]:
        old, _ = app_chunk(source, (DOCS / "pathways" / source).read_text(encoding="utf-8-sig"))
        assert not any(marker in o["text"] for o in old), f"{source} already had its intro"


# --- references --------------------------------------------------------------

def test_barcelona_now_contributes_references(c):
    _, old_refs = app_chunk(BARCELONA, (DOCS / "case_studies" / BARCELONA).read_text(encoding="utf-8-sig"))
    assert old_refs == "", "the original found none, because it only knew the underscore rule"
    new_refs = c.references[BARCELONA]
    assert len(new_refs) > 500
    assert "Treskova" in new_refs and "Montalvo" in new_refs


def test_underscore_references_still_work(c):
    assert len(c.references[WASTEWATER]) > 3000
    assert "sciencedirect" in c.references[WASTEWATER]


# --- duplicate section numbers ----------------------------------------------

DUPLICATE_DOC = (
    "1.6. Flood protection\n"
    "1.6.5. Co-benefit of safe seafood supply\n"
    "1.6.6. Co-benefit of improved food safety\n"
    "1.6.7. Trade-off of ecological disruption\n\n"
    "Flood protection is a key component of flood risk management.\n\n"
    "1.6.5. Co-benefit of safe seafood supply\n\n" + "Seafood body. " * 60 + "\n\n"
    "1.6.6. Co-benefit of improved food safety\n\n" + "Food safety body. " * 50 + "\n\n"
    "1.6.5. Trade-off of ecological disruption\n\n" + "Ecological disruption body. " * 40 + "\n\n"
    "1.6.6. Trade-off of creating habitat for disease vectors\n\n" + "Vector habitat body. " * 20 + "\n"
)


def test_a_numbering_error_no_longer_deletes_content():
    """1.6 numbers two sections twice. Keeping the longer one deleted both of
    its trade-offs while keeping both co-benefits — the exact opposite of what
    a co-benefit and trade-off tool should do."""
    old, _ = app_chunk("1.6. Flood Protection.docx.txt", DUPLICATE_DOC)
    old_titles = {o["section"] for o in old}
    assert len(old_titles) == 2, "the original collapsed four sections into two"

    c = corpus.Corpus()
    corpus.parse_pathway("1.6. Flood Protection.docx.txt", DUPLICATE_DOC, c)
    titles = {ch.section for ch in c.chunks if ch.kind == "pathway"}
    assert len(titles) == 4
    assert any("Trade-off of ecological disruption" in t for t in titles)
    assert any("Trade-off of creating habitat" in t for t in titles)
    assert any("declared 2 times with different titles" in m for _, m in c.problems)


def test_a_genuine_repeat_is_still_deduplicated():
    doc = ("2.1. Something\n\n"
           "2.1.1. A section\n\n" + "short body. " * 30 + "\n\n"
           "2.1.1. A section\n\n" + "much longer body. " * 60 + "\n")
    c = corpus.Corpus()
    corpus.parse_pathway("x.txt", doc, c)
    sections = [ch for ch in c.chunks if ch.kind == "pathway"]
    assert len({ch.section for ch in sections}) == 1
    assert "much longer body" in sections[0].text


# --- four-level headers ------------------------------------------------------

def test_four_level_headers_get_their_own_sections():
    """4.1 nests 4.1.1.1 to 4.1.1.7 under 4.1.1. The original splitter needed
    whitespace where the fourth number sits, so it never fired and ten chunks
    all carried the label 4.1.1."""
    doc = ("4.1. Green and blue spaces\n\n"
           "4.1.1. Intended outcome\n\n" + "Intro body. " * 30 + "\n\n"
           "4.1.1.1. Heat management\n\n" + "Heat body. " * 40 + "\n\n"
           "4.1.1.2. Air quality\n\n" + "Air body. " * 40 + "\n\n"
           "4.1.2. Adaptation measures\n\n" + "Measures body. " * 30 + "\n")
    old, _ = app_chunk("4.1.txt", doc)
    assert {o["section"] for o in old} == {"4.1.1 Intended outcome", "4.1.2 Adaptation measures"}

    c = corpus.Corpus()
    corpus.parse_pathway("4.1.txt", doc, c)
    sections = {ch.section for ch in c.chunks if ch.kind == "pathway"}
    assert "4.1.1.1 Heat management" in sections
    assert "4.1.1.2 Air quality" in sections
    assert len(sections) == 4


# --- glossary ----------------------------------------------------------------

def test_glossary_is_one_chunk_per_term(c):
    g = chunks_from(c, GLOSSARY)
    assert len(g) > 50
    labels = {ch.section for ch in g}
    assert len(labels) == len(g), "every term should have its own label"
    assert not any(l.startswith("Climate-Sensitive Infectious Diseases in") for l in labels), \
        "the original produced this truncated label on every chunk"


def test_glossary_terms_carry_both_definitions(c):
    lyme = [ch for ch in chunks_from(c, GLOSSARY) if ch.section.startswith("Lyme disease")]
    assert len(lyme) == 1
    assert "Borrelia" in lyme[0].text
    assert len(lyme[0].text) > 400, "should hold the short and the full definition"


def test_original_produced_one_giant_mislabelled_section():
    old, _ = app_chunk(GLOSSARY, (DOCS / "reference" / GLOSSARY).read_text(encoding="utf-8-sig"))
    assert len({o["section"] for o in old}) == 1
    assert list({o["section"] for o in old})[0] == "Climate-Sensitive Infectious Diseases in"


# --- editorial notes ---------------------------------------------------------

def test_authors_notes_are_removed_and_reported(c):
    notes = [m for m in problems_for(c, WASTEWATER) if "editorial note removed" in m]
    assert len(notes) >= 2
    text = " ".join(ch.text for ch in chunks_from(c, WASTEWATER))
    assert "Potentially best placed in another pathway" not in text
    assert "TBD - Leon" not in text


def test_the_stripper_leaves_real_prose_alone():
    kept, removed = corpus.strip_editorial([
        "NB. Potentially best placed in another pathway",
        "Nobody expects this line to be removed.",
        "TBD - Leon",
        "Notably, the risk of infectious zoonotic diseases is often linked to degradation.",
    ])
    assert len(kept) == 2 and len(removed) == 2
    assert any("Notably" in k for k in kept)


# --- manifest ----------------------------------------------------------------

def test_every_manifest_measure_exists_in_the_ontology(c):
    assert not [m for _, m in c.problems if "is not in the ontology" in m]


def test_manifest_covers_every_file_on_disk(c):
    assert not [m for _, m in c.problems if "not in the manifest" in m]


def test_missing_files_are_reported_not_fatal(c):
    missing = [s for s, m in c.problems if "not on disk" in m]
    assert len(missing) == 24, "24 of the 30 documents are not in this working tree"
    assert c.chunks, "the corpus still loaded from the six that are present"


def test_manifest_types_all_have_a_parser():
    manifest = json.loads((ROOT / "backend/data/documents.json").read_text(encoding="utf-8"))
    for entry in manifest["documents"]:
        assert entry["type"] in corpus.PARSERS
    assert set(manifest["types"]) == set(corpus.PARSERS)


def test_the_scoping_review_is_parsed_but_not_indexed(c):
    """It is matched separately and injected under its own heading. Indexing it
    as well put the same document in one context twice, the second time
    labelled as pathway evidence — and it is a quarter of the corpus by chunk
    count, so it crowded out the pathway documents it was meant to sit beside."""
    assert c.of_kind("scoping_review"), "still parsed"
    assert not any(ch.kind == "scoping_review" for ch in c.indexed())
    assert {ch.kind for ch in c.indexed()} <= corpus.INDEXED_KINDS


def test_the_glossary_and_case_studies_are_indexed(c):
    kinds = {ch.kind for ch in c.indexed()}
    assert {"glossary", "case_study", "pathway", "overview"} <= kinds


def test_publications_are_kept_out_of_the_chunk_pool(c):
    assert len(c.publications) > 40
    assert all(p.get("link") for p in c.publications)
    assert not any(ch.kind == "publications" for ch in c.chunks)


def test_citations_use_surnames():
    """The original built 'Mohammed Masud Parvage et al.' from the first name."""
    assert corpus._short_citation(
        {"title": "x", "authors": "Mohammed Masud Parvage, Jerome N. Baron, Jan C. Semenza",
         "date": "December 1, 2025"}) == "Parvage et al. (2025)"
    assert corpus._short_citation(
        {"title": "D4.1 \u2014 Something", "date": "January 2025"}) == "IDAlert D4.1 (2025)"
