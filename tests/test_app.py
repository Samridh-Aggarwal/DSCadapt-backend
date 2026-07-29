"""The HTTP surface, with the whole system faked underneath.

What matters here is the contract the frontend depends on, and the two places
the old contract was wrong: an ECDC citation for data that was never used, and
a failure returned as a successful answer.
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).parent.parent
import app as app_module  # noqa: E402
import pipeline  # noqa: E402
import prompts  # noqa: E402


# --- fakes -------------------------------------------------------------------

@dataclass
class Chunk:
    text: str = "Retention features can hold standing water."
    section: str = "1.6.6 Trade-off of habitat creation"
    source: str = "1.6. Flood Protection.docx.txt"


@dataclass
class Hit:
    chunk: object = field(default_factory=Chunk)
    score: float = 0.83
    cosine: float = 0.81
    keyword: float = 0.94
    index: int = 0


@dataclass
class Answer:
    text: str = "Retention basins may create standing water that could support Culex breeding."
    route: str = "both"
    rewritten: str = "what are the disease risks of retention basins in Italy"
    in_scope: bool = True
    router_parsed: bool = True
    hits: list = field(default_factory=lambda: [Hit()])
    references: dict = field(default_factory=dict)
    web_text: str = "Recent reporting on West Nile virus."
    web_failed: bool = False
    thin_coverage: bool = False
    diseases: list = field(default_factory=lambda: ["West Nile fever"])
    ecdc_block: str = "ECDC SURVEILLANCE DATA FOR ITALY (2018-2024)"
    publications: list = field(default_factory=list)
    scoping: list = field(default_factory=list)
    context_chars: int = 3000
    violations: list = field(default_factory=list)
    timing: dict = field(default_factory=lambda: {"total": 2.5})
    usage: dict = field(default_factory=dict)

    @property
    def used_surveillance(self):
        return bool(self.ecdc_block)


class FakePipeline:
    def __init__(self, answer=None, error=None):
        self.answer, self.error, self.calls = answer or Answer(), error, []

    def respond(self, message, history=None, audience="Policymaker",
                country="All Europe", length="Standard"):
        self.calls.append({"message": message, "audience": audience,
                           "country": country, "length": length})
        if self.error:
            raise self.error
        return self.answer


class FakeTranslator:
    def __init__(self, available=True):
        self.available, self.calls = available, []

    def to_english(self, text):
        self.calls.append(("in", text))
        return f"EN({text})"

    def from_english(self, text, target):
        self.calls.append(("out", target))
        return f"{target}({text})"

    def answer(self, text, target):
        self.calls.append(("answer", target))
        return f"{target}({text})"


class FakeLogger:
    version = "test"

    def __init__(self):
        self.records = []

    def record(self, entry):
        self.records.append(entry)

    def close(self):
        pass

    def health(self):
        return {"written": len(self.records), "sink": "NullSink", "version": "test"}

    def kinds(self):
        return [r.get("kind") for r in self.records]


class FakeSurveillance:
    available = True
    year_range = "2018-2024"
    years = (2018, 2024)

    def country_totals(self, disease):
        return [("Italy", 900), ("Greece", 300)]

    def known_diseases(self):
        return ["West Nile"]

    def has_country(self, c):
        return c != "United Kingdom"

    def relevant(self, d, c):
        return True


@dataclass
class FakeSettings:
    log_text: bool = True


@dataclass
class FakeServices:
    pipeline: object
    translator: object
    log: object
    settings: object = field(default_factory=FakeSettings)
    surveillance: object = field(default_factory=FakeSurveillance)
    index: object = field(default_factory=lambda: [1] * 309)
    corpus: object = None
    reviewer: object = None
    chat: object = None
    embed: object = None
    web_agent_id: str = "agent"

    def health(self):
        return {"status": "ok", "chunks": 309, "web_search": True, "deepl": True,
                "ecdc": True, "ecdc_years": "2018-2024", "version": "test"}


def build(answer=None, error=None, translator=None):
    services = FakeServices(pipeline=FakePipeline(answer, error),
                            translator=translator or FakeTranslator(),
                            log=FakeLogger())
    return TestClient(app_module.create_app(services)), services


# --- health ------------------------------------------------------------------

def test_health_reports_the_parts_not_just_that_the_process_is_alive():
    client, _ = build()
    body = client.get("/").json()
    assert body["status"] == "ok"
    for field_ in ["chunks", "web_search", "deepl", "ecdc", "ecdc_years", "version", "logger"]:
        assert field_ in body


# --- ask ---------------------------------------------------------------------

def test_an_english_question_is_not_translated():
    client, services = build()
    body = client.post("/ask", json={"question": "wetlands in Valencia"}).json()
    assert services.translator.calls == []
    assert body["body"] and body["meta"]["route"] == "both"


def test_a_non_english_question_goes_through_the_sandwich():
    client, services = build()
    body = client.post("/ask", json={"question": "zones humides",
                                     "language": "Français"}).json()
    directions = [d for d, _ in services.translator.calls]
    assert "in" in directions and "answer" in directions
    assert services.pipeline.calls[0]["message"].startswith("EN(")
    assert body["body"].startswith("FR(")


def test_the_english_original_comes_back_alongside_the_translation():
    """So the frontend can store it as history instead of the translated text,
    which the model then reads while being told not to mirror the language."""
    client, _ = build()
    body = client.post("/ask", json={"question": "q", "language": "Deutsch"}).json()
    assert body["english_answer"] == Answer().text
    assert body["english_answer"] != body["body"]


def test_an_english_answer_is_highlighted():
    client, _ = build()
    body = client.post("/ask", json={"question": "wetlands"}).json()
    assert "english_answer" in body


def test_a_translated_answer_is_not_highlighted():
    """Highlighting matches English source text, so it would find nothing."""
    client, _ = build()
    body = client.post("/ask", json={"question": "q", "language": "Italiano"}).json()
    assert "<span" not in body["body"]


def test_the_sources_panel_carries_the_retrieved_sections():
    client, _ = build()
    sources = client.post("/ask", json={"question": "wetlands"}).json()["sources"]
    evidence = [s for s in sources if s["type"] == "evidence"]
    assert evidence and evidence[0]["title"] == "1.6.6 Trade-off of habitat creation"
    assert ".docx.txt" not in evidence[0]["origin"]


def test_an_ecdc_citation_appears_only_when_data_was_used():
    """It was built from the disease list, which is populated even when the
    block is empty — so a UK user was shown a citation to data that never was."""
    client, _ = build(Answer())
    sources = client.post("/ask", json={"question": "q", "country": "Italy"}).json()["sources"]
    assert any(s["type"] == "ecdc" for s in sources)

    detected_but_unused = Answer(ecdc_block="", diseases=["West Nile fever"])
    client, _ = build(detected_but_unused)
    body = client.post("/ask", json={"question": "q", "country": "United Kingdom"}).json()
    assert not any(s["type"] == "ecdc" for s in body["sources"])


def test_a_redirect_returns_the_message_and_the_route():
    client, _ = build(Answer(route="redirect", in_scope=False, text=prompts.REDIRECT,
                             hits=[], web_text="", ecdc_block="", diseases=[]))
    body = client.post("/ask", json={"question": "how do I cook pasta"}).json()
    assert body["meta"]["route"] == "redirect"
    assert body["body"] == prompts.REDIRECT
    assert body["sources"] == []


def test_a_failure_is_a_502_not_an_answer():
    """It used to be caught and returned as the answer text at HTTP 200, so the
    user saw 'Error: 429 rate limit exceeded' and monitoring saw a healthy
    response."""
    client, services = build(error=pipeline.PipelineError("generation failed: 429"))
    response = client.post("/ask", json={"question": "wetlands"})
    assert response.status_code == 502
    assert "429" not in response.text, "the upstream error should not reach the user"
    assert services.log.kinds() == ["error"]


def test_every_answer_is_logged_once():
    client, services = build()
    client.post("/ask", json={"question": "wetlands", "country": "Italy",
                              "language": "Français", "length": "Brief"})
    assert len(services.log.records) == 1
    record = services.log.records[0]
    assert record["settings"]["country"] == "Italy"
    assert record["settings"]["language"] == "Français"
    assert record["prompt_sha"]


def test_the_turn_number_counts_user_messages():
    client, services = build()
    client.post("/ask", json={"question": "third", "history": [
        {"role": "user", "content": "first"}, {"role": "assistant", "content": "a"},
        {"role": "user", "content": "second"}, {"role": "assistant", "content": "b"}]})
    assert services.log.records[0]["turn"] == 3


def test_settings_reach_the_pipeline():
    client, services = build()
    client.post("/ask", json={"question": "q", "audience": "Researcher",
                              "country": "Spain", "length": "Detailed"})
    call = services.pipeline.calls[0]
    assert call["audience"] == "Researcher" and call["country"] == "Spain"
    assert call["length"] == "Detailed"


# --- documents ---------------------------------------------------------------

def test_a_transcript_with_no_conversation_is_a_400():
    client, _ = build()
    assert client.post("/transcript", json={"history": []}).status_code == 400


def test_a_briefing_needs_enough_conversation():
    client, _ = build()
    response = client.post("/briefing", json={"history": [
        {"role": "user", "content": "q"},
        {"role": "assistant", "route": "domain", "content": "a"}]})
    assert response.status_code in (400, 502)


# --- reviewer ----------------------------------------------------------------

class FakeReviewer:
    def __init__(self, result=None, error=None):
        self.result, self.error = result, error

    def review(self, text, **kwargs):
        if self.error:
            raise self.error
        return self.result


@dataclass
class FakeReview:
    note: str = "Selective review, 1 consideration found."
    comments: str = "PASSAGE: basins\nCONSIDERATION: may support West Nile fever."
    shortlist: list = field(default_factory=lambda: ["1.6. Flood Protection"])
    out_of_scope: bool = False

    def log_record(self, country, text, include_comments=True):
        record = {"country": country, "document_chars": len(text), "document_sha1": "abc123"}
        if include_comments:
            record["comments"] = self.comments
        return record


def test_a_review_returns_its_comments_and_logs_no_document_text():
    services = FakeServices(pipeline=FakePipeline(), translator=FakeTranslator(),
                            log=FakeLogger(), reviewer=FakeReviewer(FakeReview()))
    client = TestClient(app_module.create_app(services))
    body = client.post("/evaluate", json={"text": "Our plan builds retention basins.",
                                          "country": "Germany"}).json()
    assert "West Nile" in body["comments"]
    assert "retention basins" not in str(services.log.records)


def test_debug_disease_is_available_when_debug_routes_are_on(monkeypatch):
    monkeypatch.setenv("EXPOSE_DEBUG_ROUTES", "1")
    services = FakeServices(pipeline=FakePipeline(), translator=FakeTranslator(),
                            log=FakeLogger())
    client = TestClient(app_module.create_app(services))
    assert client.get("/debug/disease?disease=West+Nile+fever").status_code == 200


def test_the_briefing_log_holds_the_draft_as_well_as_the_final():
    """The difference between them is what the audit removed, which is the only
    way to tell whether the audit does anything."""
    import render
    class Chat:
        def __call__(self, model, messages, **kw):
            import prompts as pr
            text = ("### SUMMARY\nS\n### CO-BENEFITS\nC\n### TRADE-OFFS\n"
                    + ("T" if messages[0]["content"] is pr.BRIEFING_AUDIT else "T and a ranking")
                    + "\n### UNCERTAINTIES\nU")
            return type("C", (), {"text": text, "usage": None})()

    services = FakeServices(pipeline=FakePipeline(), translator=FakeTranslator(),
                            log=FakeLogger(), chat=Chat())
    client = TestClient(app_module.create_app(services))
    client.post("/briefing", json={"history": [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "route": "both", "content": "a1"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "route": "domain", "content": "a2"}]})
    entry = [r for r in services.log.records if r.get("kind") == "briefing"][0]
    assert "a ranking" in entry["draft"] and "a ranking" not in entry["final"]


def test_a_document_the_reviewer_rejects_is_a_400():
    services = FakeServices(pipeline=FakePipeline(), translator=FakeTranslator(),
                            log=FakeLogger(),
                            reviewer=FakeReviewer(error=ValueError("too long")))
    client = TestClient(app_module.create_app(services))
    response = client.post("/evaluate", json={"text": "x"})
    assert response.status_code == 400 and "too long" in response.text


# --- extraction --------------------------------------------------------------

def test_a_text_upload_comes_back_as_text():
    client, _ = build()
    files = {"file": ("plan.txt", b"Our adaptation plan.", "text/plain")}
    body = client.post("/extract", files=files).json()
    assert body["text"] == "Our adaptation plan." and body["chars"] == 20


def test_an_unsupported_type_is_refused():
    client, _ = build()
    body = client.post("/extract", files={"file": ("x.xlsx", b"data", "application/x")}).json()
    assert "Unsupported file type" in body["error"]


def test_an_empty_file_says_so():
    client, _ = build()
    body = client.post("/extract", files={"file": ("x.txt", b"   ", "text/plain")}).json()
    assert "No readable text" in body["error"]


def test_a_document_too_long_for_the_reviewer_is_flagged_at_upload():
    """Rather than accepted here and rejected later."""
    client, _ = build()
    big = b"a" * 70_000
    body = client.post("/extract", files={"file": ("x.txt", big, "text/plain")}).json()
    assert "warning" in body and "reviewer takes" in body["warning"]


# --- gatekeeping -------------------------------------------------------------

def test_an_api_key_is_required_only_when_one_is_set(monkeypatch):
    monkeypatch.setenv("API_KEY", "secret")
    services = FakeServices(pipeline=FakePipeline(), translator=FakeTranslator(),
                            log=FakeLogger())
    client = TestClient(app_module.create_app(services))
    assert client.post("/ask", json={"question": "q"}).status_code == 401
    assert client.post("/ask", json={"question": "q"},
                       headers={"X-API-Key": "secret"}).status_code == 200


def test_the_rate_limit_is_off_unless_configured(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT", "2")
    services = FakeServices(pipeline=FakePipeline(), translator=FakeTranslator(),
                            log=FakeLogger())
    client = TestClient(app_module.create_app(services))
    assert client.post("/ask", json={"question": "q"}).status_code == 200
    assert client.post("/ask", json={"question": "q"}).status_code == 200
    assert client.post("/ask", json={"question": "q"}).status_code == 429


def test_debug_routes_are_absent_by_default():
    client, _ = build()
    assert client.get("/debug/corpus").status_code == 404


def test_the_dead_pdf_endpoint_is_gone():
    """generate_pdf never called pdf.output() and never returned a path, so
    /pdf raised on every call. fpdf2 was a dependency for nothing else."""
    client, _ = build()
    assert client.post("/pdf", json={"history": []}).status_code == 404
