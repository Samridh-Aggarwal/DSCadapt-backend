"""The reviewer, the highlighter and the translator.

The reviewer's graph walk is the part worth guarding: it is what stops the
model naming a disease that is not established where the reader is.
"""

import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
import attribution  # noqa: E402
import corpus  # noqa: E402
import ecdc  # noqa: E402
import ontology  # noqa: E402
import retrieval  # noqa: E402
import review  # noqa: E402
import translate  # noqa: E402

RNG = np.random.default_rng(4242)
DIM = 48
COUNTRIES = ["Germany", "Italy", "Spain", "France", "Greece", "Finland"]


@dataclass
class Completion:
    text: str
    usage: dict = None


class FakeChat:
    def __init__(self, scope="IN", comments="NO GROUNDED CONSIDERATIONS", fail=None):
        self.calls, self.scope, self.comments, self.fail = [], scope, comments, fail

    def __call__(self, model, messages, **kw):
        self.calls.append({"model": model, "messages": messages, **kw})
        if self.fail:
            raise RuntimeError(self.fail)
        return Completion(self.scope if model == review.SCOPE_MODEL else self.comments)

    @property
    def review_message(self):
        for call in reversed(self.calls):
            if call["model"] == review.REVIEW_MODEL:
                return call["messages"][-1]["content"]
        return None


def fake_embed(texts):
    return RNG.normal(size=(len(texts), DIM))


def stable_embed(texts):
    """Deterministic: the same string always gets the same vector.

    Needed wherever a test asserts that similar text scores as similar. Random
    vectors cannot express that, so a sentence copied word for word from its
    source scored near zero.
    """
    out = []
    for text in texts:
        seed = int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:8], 16)
        out.append(np.random.default_rng(seed).normal(size=DIM))
    return np.array(out)


# Three sources that do map to measures carrying trade-off edges. The corpus in
# this working tree holds six documents and only one of them qualifies, which
# is a property of the fixtures rather than of the reviewer.
REVIEWABLE = {
    "1.6. Flood Protection.docx.txt": "Flood protection",
    "4.1. Increasing green and blue spaces.docx.txt": "Increase green & blue spaces",
    "3.4. Expand forested and natural areas.docx.txt": "Expand forested and natural areas",
}


@dataclass
class Chunk:
    text: str
    source: str
    section: str
    kind: str = "pathway"


@pytest.fixture(scope="module")
def parts():
    chunks = []
    for source in REVIEWABLE:
        stem = source.split(".")[0] + "." + source.split(".")[1]
        for n in range(4):
            chunks.append(Chunk(
                f"{stem}.{n} Retention and habitat. Standing water in engineered features can "
                f"support mosquito breeding and raise transmission risk. " * 3,
                source, f"{stem}.{n} Trade-off"))
    index = retrieval.Index(chunks, stable_embed([c.text for c in chunks]))

    rows = [{"disease": d["ecdc"], "country": k, "year": y, "cases": 90, "deaths": None,
             "group": d["group"], "dark_figure": "High"}
            for d in ecdc.DISEASES.values() if d["ecdc"]
            for k in COUNTRIES for y in (2023, 2024)]
    return REVIEWABLE, index, ecdc.Surveillance(pd.DataFrame(rows), loaded_at=0)


def make(parts, chat, embed=None):
    measures, index, surveillance = parts
    return review.Reviewer(index, surveillance, measures, embed or stable_embed, chat)


FLOOD_DOC = ("Our city is investing in flood protection infrastructure.\n\n"
             "New retention basins will be built across the eastern districts to hold "
             "stormwater during heavy rainfall events and release it slowly.\n\n"
             "The programme will also restore wetland areas along the river corridor.\n")


# --- reviewer ----------------------------------------------------------------

def test_a_document_that_matches_nothing_is_reported_cleanly(parts):
    r = make(parts, FakeChat()).review(FLOOD_DOC, country="Germany", min_score=0.99)
    assert r.note == review.NO_MATCH
    assert r.comments == "NO GROUNDED CONSIDERATIONS"


def test_an_out_of_scope_document_stops_before_matching(parts):
    chat = FakeChat(scope="OUT")
    r = make(parts, chat).review(FLOOD_DOC, country="Germany")
    assert r.out_of_scope and r.note == review.OUT_OF_SCOPE
    assert len(chat.calls) == 1, "no review call for a rejected document"


def test_a_failed_scope_check_accepts_but_records_that_it_did_not_run(parts):
    class Flaky(FakeChat):
        def __call__(self, model, messages, **kw):
            if model == review.SCOPE_MODEL:
                raise RuntimeError("timeout")
            return super().__call__(model, messages, **kw)

    r = make(parts, Flaky()).review(FLOOD_DOC, country="Germany", min_score=-1.0)
    assert not r.out_of_scope
    assert not r.scope_checked


def test_only_measure_bearing_documents_reach_the_shortlist(parts):
    """The glossary and the scoping review can never produce a comment, so
    counting them filled shortlist slots the graph walk then discarded."""
    measures, index, surveillance = parts
    extra = [Chunk("A glossary entry about West Nile fever. " * 20,
                   "DST_disease_summaries.docx.txt", "West Nile fever")]
    wider = retrieval.Index(list(index.chunks) + extra,
                            stable_embed([c.text for c in index.chunks] + [extra[0].text]))
    r = review.Reviewer(wider, surveillance, measures, stable_embed, FakeChat())
    result = r.review(FLOOD_DOC, country="Germany", min_score=-1.0)
    assert "DST_disease_summaries" not in " ".join(result.shortlist)
    assert result.shortlist


def test_the_measure_block_carries_the_filtered_disease_list(parts):
    chat = FakeChat()
    make(parts, chat).review(FLOOD_DOC, country="Germany", min_score=-1.0)
    message = chat.review_message
    assert "MEASURE:" in message
    assert "DISEASES TO CONSIDER (already filtered for Germany)" in message
    assert "HOW IT CAN RAISE RISK:" in message
    assert "EVIDENCE:" in message


def test_the_allowlist_is_the_graph_walk_filtered_by_country(parts):
    _, _, surveillance = parts
    r = make(parts, FakeChat()).review(FLOOD_DOC, country="Germany", min_score=-1.0)
    for measure in r.measures:
        for disease in ontology.diseases_for(measure):
            if surveillance.relevant(disease, "Germany"):
                assert disease in r.allowed
            else:
                assert disease not in r.allowed


def test_a_comment_naming_a_disease_outside_the_list_is_discarded(parts):
    chat = FakeChat(comments=(
        "PASSAGE: retention basins\n"
        "CONSIDERATION: standing water may support Culex breeding and West Nile fever "
        "transmission, which the document does not appear to address.\n\n"
        "PASSAGE: wetland areas\n"
        "CONSIDERATION: this could raise the risk of Ebola, which the document does not "
        "appear to address."))
    r = make(parts, chat).review(FLOOD_DOC, country="Germany", min_score=-1.0)
    assert r.n_comments == 1
    assert "West Nile" in r.comments and "Ebola" not in r.comments
    assert len(r.dropped) == 1


def test_prose_around_the_comments_is_thrown_away(parts):
    chat = FakeChat(comments=(
        "Here are my thoughts on the document.\n\n"
        "PASSAGE: retention basins\n"
        "CONSIDERATION: may support West Nile fever transmission.\n\n"
        "Overall this is a strong plan."))
    r = make(parts, chat).review(FLOOD_DOC, country="Germany", min_score=-1.0)
    assert "Overall this is a strong plan" not in r.comments
    assert r.n_comments == 1


def test_no_grounded_considerations_passes_through(parts):
    r = make(parts, FakeChat()).review(FLOOD_DOC, country="Germany", min_score=-1.0)
    assert r.comments == "NO GROUNDED CONSIDERATIONS" and r.n_comments == 0
    assert "0 considerations found" in r.note


def test_disease_surface_forms_cover_the_ways_a_name_is_written():
    forms = review.surface_forms("Crimean-Congo haemorraghic fever (CCHF)")
    assert "cchf" in forms
    assert "crimean-congo haemorraghic fever (cchf)" in forms
    assert "crimean-congo" in forms
    assert "west nile" in review.surface_forms("West Nile fever")


def test_names_allowed_matches_an_alias_not_just_the_full_name():
    assert review.names_allowed("risk of legionella in the system", ["Legionnaires' disease"])
    assert not review.names_allowed("risk of measles", ["Legionnaires' disease"])


def test_an_empty_or_oversized_document_is_rejected(parts):
    r = make(parts, FakeChat())
    with pytest.raises(ValueError):
        r.review("   ")
    with pytest.raises(ValueError, match="too long"):
        r.review("x" * (review.MAX_CHARS + 1))


def test_document_paragraphs_are_embedded_in_batches(parts):
    """A sixty-thousand-character document is two hundred-odd paragraphs, and
    the original sent all of them in one request."""
    sizes = []

    def counting_embed(texts):
        sizes.append(len(texts))
        return RNG.normal(size=(len(texts), DIM))

    measures, index, surveillance = parts
    r = review.Reviewer(index, surveillance, measures, counting_embed, FakeChat())
    long_doc = "\n\n".join(f"Paragraph {i} about flood protection and retention." * 3
                           for i in range(250))
    r.review(long_doc, country="Germany", min_score=0.99)
    assert max(sizes) <= retrieval.EMBED_BATCH
    assert sum(s for s in sizes if s > 1) >= 250


def test_the_log_record_holds_no_document_text(parts):
    """Someone reviewing an unpublished strategy should not find it in a dataset."""
    r = make(parts, FakeChat()).review(FLOOD_DOC, country="Germany", min_score=-1.0)
    record = r.log_record("Germany", FLOOD_DOC)
    blob = str(record)
    assert "retention basins" not in blob
    assert "flood protection infrastructure" not in blob
    assert record["document_chars"] == len(FLOOD_DOC)
    assert len(record["document_sha1"]) == 16


def test_the_uk_now_reaches_diseases_it_could_not_before(parts):
    """The missing-country fix landing in the reviewer."""
    _, _, surveillance = parts
    uk = {d for d in ecdc.DISEASES if surveillance.relevant(d, "United Kingdom")}
    for disease in ["Legionnaires' disease", "Leptospirosis", "Hantavirus infection"]:
        assert disease in uk


# --- attribution -------------------------------------------------------------

@dataclass
class FakeHit:
    chunk: object


@dataclass
class FakeChunk:
    text: str
    section: str = "1.6.6 Trade-off"
    source: str = "1.6.txt"


def test_sentences_split_on_punctuation_and_paragraphs():
    assert attribution.sentences("One. Two!\n\nThree?") == ["One.", "Two!", "Three?"]


def test_short_sentences_are_skipped_and_counted():
    a = attribution.annotate("Yes. " + "A long enough sentence to be scored properly. " * 3,
                             hits=[FakeHit(FakeChunk("Some source text here."))],
                             embed=stable_embed)
    assert a.skipped_short >= 1


def test_surveillance_figures_are_matched_lexically():
    block = "West Nile in Italy\n  Reported cases: 2023: 332, 2024: 460\n"
    found = attribution.surveillance_phrases(block, "There were 332 cases reported in 2023.")
    assert any("332" in f for f in found)


def test_a_figure_written_in_words_is_not_matched():
    """A real limitation, pinned so it is not mistaken for a passing test."""
    block = "West Nile in Italy\n  Reported cases: 2023: 332\n"
    assert attribution.surveillance_phrases(block, "There were just over three hundred cases") == []


def test_tiny_numbers_are_ignored():
    block = "Dengue in Spain\n  Reported cases: 2023: 3\n"
    assert attribution.surveillance_phrases(block, "There were 3 cases.") == []


@pytest.mark.parametrize("domain,web,expected", [
    (0.90, 0.50, "domain"),
    (0.50, 0.95, "web"),
    (0.50, 0.50, None),
    (0.90, 0.93, "domain"),    # web qualifies but not by the margin
    (0.88, 0.96, "web"),       # web clears the margin
])
def test_the_classification_rule(domain, web, expected):
    assert attribution.classify(domain, web) == expected


def test_surveillance_wins_outright_over_scoring():
    block = "West Nile in Italy\n  Reported cases: 2023: 332\n"
    text = "Surveillance recorded 332 cases in 2023 across the northern regions."
    a = attribution.annotate(text, hits=[FakeHit(FakeChunk(text))],
                             surveillance_block=block, embed=stable_embed)
    assert "ecdc-highlight" in a.html
    assert a.counts.get("surveillance") == 1


def test_identical_text_scores_as_domain():
    claim = "Retention basins can create standing water that supports Culex breeding."
    a = attribution.annotate(claim, hits=[FakeHit(FakeChunk(claim))], embed=stable_embed)
    assert "highlight" in a.html


def test_the_reference_list_is_never_highlighted():
    body = "Retention basins can create standing water that supports Culex breeding."
    text = body + "\n\nReferences\nSemenza, J. 2018. Something."
    a = attribution.annotate(text, hits=[FakeHit(FakeChunk(body))], embed=stable_embed)
    assert "<span" not in a.html.split("References")[1]


def test_a_scoring_failure_returns_the_answer_unhighlighted():
    def broken(texts):
        raise RuntimeError("embedding down")

    text = "A sentence long enough to be scored by the attribution step here."
    a = attribution.annotate(text, hits=[FakeHit(FakeChunk("source"))], embed=broken)
    assert a.html == text and a.counts == {}


def test_no_embedder_means_no_attribution():
    assert attribution.annotate("text", embed=None).html == "text"


def test_the_summary_reports_what_happened():
    a = attribution.annotate("Retention basins can create standing water for Culex breeding.",
                             hits=[FakeHit(FakeChunk("Retention basins create standing water."))],
                             embed=stable_embed)
    assert "embeddings" in a.summary()


# --- translation -------------------------------------------------------------

class FakeDeepL:
    def __init__(self):
        self.calls = []

    def translate_text(self, text, target_lang):
        self.calls.append((text, target_lang))
        return type("R", (), {"text": f"[{target_lang}] {text}"})()


def test_english_is_a_passthrough():
    assert translate.code_for("English") is None
    assert translate.code_for("Deutsch") == "DE"


def test_no_client_means_everything_passes_through():
    t = translate.Translator(None)
    assert not t.available
    assert t.to_english("bonjour") == "bonjour"
    assert t.answer("text", "FR") == "text"


def test_the_reference_list_is_split_off_before_translating():
    """Citations translated as prose come back with author names localised."""
    t = translate.Translator(FakeDeepL())
    out = t.answer("The body of the answer.\n\nReferences\nSemenza, J.C. (2018). A paper.", "FR")
    assert "Semenza, J.C. (2018). A paper." in out
    assert "[FR] The body" in out
    assert "[FR] References" in out


def test_an_answer_with_no_references_is_translated_whole():
    t = translate.Translator(FakeDeepL())
    assert t.answer("Just the body.", "FR") == "[FR] Just the body."


def test_a_translation_failure_returns_the_original():
    class Broken:
        def translate_text(self, text, target_lang):
            raise RuntimeError("quota exceeded")

    t = translate.Translator(Broken())
    assert t.from_english("hello", "DE") == "hello"


def test_characters_are_counted_for_cost():
    t = translate.Translator(FakeDeepL())
    t.to_english("bonjour le monde")
    assert t.characters == len("bonjour le monde")


def test_every_language_the_interface_offers_has_a_code():
    ui = ["English", "Čeština", "Dansk", "Deutsch", "Eesti", "Español", "Français", "Gaeilge",
          "Hrvatski", "Íslenska", "Italiano", "Latviešu", "Lietuvių", "Malti", "Magyar",
          "Nederlands", "Norsk", "Polski", "Português", "Română", "Slovenčina", "Slovenščina",
          "Suomi", "Svenska", "Български", "Ελληνικά"]
    assert set(ui) == set(translate.LANGUAGES)
    assert all(v for k, v in translate.LANGUAGES.items() if k != "English")
