"""Traces, wiring and documents.

The logger tests care about two things: that recording is cheap and off the
request path, and that a trace holds enough to be a golden-set entry.
"""

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
import config  # noqa: E402
import logger as lg  # noqa: E402
import prompts  # noqa: E402
import render  # noqa: E402


# --- stand-ins ---------------------------------------------------------------

@dataclass
class FakeHit:
    chunk: object
    score: float = 0.81
    cosine: float = 0.79
    keyword: float = 0.92
    index: int = 0


@dataclass
class FakeChunk:
    text: str = "body"
    section: str = "1.6.6 Trade-off of habitat creation"
    source: str = "1.6. Flood Protection.docx.txt"


@dataclass
class FakeViolation:
    rule: str
    detail: str
    advisory: bool = False


@dataclass
class FakeAnswer:
    text: str = "An answer about wetlands and mosquito habitat."
    route: str = "both"
    rewritten: str = "what are the disease risks of wetlands in Valencia"
    in_scope: bool = True
    router_parsed: bool = True
    hits: list = field(default_factory=lambda: [FakeHit(FakeChunk())])
    references: dict = field(default_factory=dict)
    web_text: str = "Some web text."
    web_failed: bool = False
    thin_coverage: bool = False
    diseases: list = field(default_factory=lambda: ["West Nile fever"])
    ecdc_block: str = "ECDC SURVEILLANCE DATA FOR SPAIN"
    publications: list = field(default_factory=list)
    scoping: list = field(default_factory=list)
    context_chars: int = 4200
    violations: list = field(default_factory=list)
    timing: dict = field(default_factory=lambda: {"router": 0.4, "retrieval": 0.1,
                                                  "generation": 3.2, "total": 4.1})
    usage: dict = field(default_factory=lambda: {"router": {"input": 300, "output": 40},
                                                 "generation": {"input": 6000, "output": 500}})

    @property
    def used_surveillance(self):
        return bool(self.ecdc_block)


class RecordingSink:
    def __init__(self, fail=False):
        self.sends, self.fail = [], fail

    def send(self, path):
        if self.fail:
            raise RuntimeError("hub unreachable")
        self.sends.append(Path(path).read_text(encoding="utf-8"))


def lines_in(directory):
    out = []
    for path in Path(directory).rglob("*.jsonl"):
        out.extend(json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l)
    return out


# --- transport ---------------------------------------------------------------

def test_recording_writes_a_line_and_does_not_upload(tmp_path):
    sink = RecordingSink()
    log = lg.Logger(tmp_path, sink=sink, batch=10, flush_seconds=999)
    log.record({"kind": "ask", "route": "domain"})
    assert len(lines_in(tmp_path)) == 1
    assert sink.sends == [], "the upload is the background thread's job"
    log.close()


def test_recording_is_fast(tmp_path):
    """It used to be two hub uploads and a full HTML rebuild, inside the request."""
    log = lg.Logger(tmp_path, sink=RecordingSink(), batch=10_000, flush_seconds=999)
    started = time.perf_counter()
    for i in range(200):
        log.record({"kind": "ask", "n": i})
    elapsed = time.perf_counter() - started
    assert elapsed < 0.5, f"{elapsed*1000:.0f} ms for 200 records"
    assert len(lines_in(tmp_path)) == 200
    log.close()


def test_a_full_batch_triggers_an_upload(tmp_path):
    sink = RecordingSink()
    log = lg.Logger(tmp_path, sink=sink, batch=3, flush_seconds=999)
    for i in range(3):
        log.record({"n": i})
    for _ in range(50):
        if sink.sends:
            break
        time.sleep(0.02)
    assert sink.sends, "the worker should have uploaded"
    assert sink.sends[0].count("\n") == 3
    log.close()


def test_an_upload_failure_keeps_the_local_file(tmp_path):
    log = lg.Logger(tmp_path, sink=RecordingSink(fail=True), batch=1, flush_seconds=999)
    log.record({"n": 1})
    log.flush()
    assert len(lines_in(tmp_path)) == 1
    assert log.failures >= 1
    log.close()


def test_a_broken_record_never_reaches_the_caller(tmp_path):
    log = lg.Logger(tmp_path, sink=RecordingSink())
    class Awkward:
        def __repr__(self):
            raise RuntimeError("nope")
    log.record({"weird": Awkward()})   # must not raise
    log.close()


def test_each_process_writes_its_own_file(tmp_path):
    """No read-modify-write, so two instances cannot overwrite each other."""
    a, b = lg.Logger(tmp_path, batch=99), lg.Logger(tmp_path, batch=99)
    a.record({"from": "a"})
    b.record({"from": "b"})
    files = list(Path(tmp_path).rglob("*.jsonl"))
    assert len(files) == 2 and a.boot != b.boot
    a.close(); b.close()


def test_the_default_sink_publishes_nothing(tmp_path):
    log = lg.Logger(tmp_path)
    assert isinstance(log.sink, lg.NullSink)
    log.record({"n": 1})
    log.flush()
    log.close()


def test_health_reports_the_state(tmp_path):
    log = lg.Logger(tmp_path, sink=RecordingSink())
    log.record({"n": 1})
    health = log.health()
    assert health["written"] == 1 and health["sink"] == "RecordingSink"
    assert health["version"] and health["boot"]
    log.close()


# --- what a trace holds ------------------------------------------------------

def test_the_settings_that_were_never_captured():
    """Without these, traffic cannot be sliced by country, and country is the
    axis the surveillance behaviour turns on."""
    record = lg.query_record("q", FakeAnswer(), audience="Researcher", country="Spain",
                             language="Español", length="Detailed")
    assert record["settings"] == {"audience": "Researcher", "country": "Spain",
                                  "language": "Español", "length": "Detailed"}


def test_retrieval_scores_are_split_into_their_halves():
    """Only the blended number was stored, so 0.85/0.15 could not be tuned from
    real queries."""
    hit = lg.query_record("q", FakeAnswer())["retrieval"]["hits"][0]
    assert {"score", "cosine", "keyword"} <= set(hit)
    assert hit["cosine"] != hit["keyword"]


def test_a_redirect_is_visible_as_a_redirect():
    answer = FakeAnswer(route="redirect", in_scope=False, hits=[])
    record = lg.query_record("how do I cook pasta", answer)
    assert record["redirected"] and record["router"]["route"] == "redirect"


def test_an_unparsed_router_is_visible():
    record = lg.query_record("q", FakeAnswer(router_parsed=False))
    assert record["router"]["parsed"] is False
    assert "ROUTER UNPARSED" in lg.summarise(record)


def test_the_prompt_fingerprint_moves_when_the_prompt_does():
    a = lg.query_record("q", FakeAnswer(), system_prompt="one")["prompt_sha"]
    b = lg.query_record("q", FakeAnswer(), system_prompt="two")["prompt_sha"]
    assert a != b and len(a) == 12


def test_surveillance_use_is_recorded_not_inferred():
    used = lg.query_record("q", FakeAnswer())
    assert used["surveillance"]["used"]
    unused = lg.query_record("q", FakeAnswer(ecdc_block=""))
    assert not unused["surveillance"]["used"]
    assert unused["surveillance"]["diseases"], "diseases detected but no data injected"


def test_violations_are_carried_into_the_trace():
    answer = FakeAnswer(violations=[FakeViolation("recommendation", "you should"),
                                    FakeViolation("length", "620 words", advisory=True)])
    record = lg.query_record("q", answer)
    assert len(record["violations"]) == 2
    assert "violations: recommendation" in lg.summarise(record)


def test_cost_is_computed_per_stage_and_totalled():
    record = lg.query_record("q", FakeAnswer())
    assert record["cost"]["router"] > 0 and record["cost"]["generation"] > 0
    assert record["cost"]["total"] == pytest.approx(
        record["cost"]["router"] + record["cost"]["generation"])


def test_timing_survives_per_stage():
    assert set(lg.query_record("q", FakeAnswer())["timing"]) == {
        "router", "retrieval", "generation", "total"}


def test_question_text_can_be_switched_off():
    """It is what makes a trace a golden-set entry, and it is also the part a
    consent notice has to cover."""
    on = lg.query_record("what about Valencia", FakeAnswer())
    assert on["message"] == "what about Valencia" and "answer" in on
    off = lg.query_record("what about Valencia", FakeAnswer(), capture_text=False)
    assert "message" not in off and "answer" not in off
    assert off["message_chars"] == len("what about Valencia")
    assert off["router"]["rewritten"] is None


def test_an_error_record_holds_no_document_text():
    record = lg.error_record("a long question", "generation", RuntimeError("429"), country="Italy")
    assert record["error"]["type"] == "RuntimeError"
    assert "a long question" not in str(record)
    assert "ERROR generation" in lg.summarise(record)


def test_the_trace_carries_everything_a_golden_set_entry_needs():
    record = lg.query_record("what are the risks", FakeAnswer(), country="Spain")
    for field_ in ["message", "answer", "settings", "router", "retrieval", "surveillance"]:
        assert field_ in record
    assert record["retrieval"]["sources"], "which documents answered it"


# --- config ------------------------------------------------------------------

def test_env_casts_and_falls_back(monkeypatch):
    monkeypatch.setenv("X_NUM", "7")
    assert config.env("X_NUM", 0, int) == 7
    monkeypatch.setenv("X_NUM", "seven")
    assert config.env("X_NUM", 3, int) == 3
    monkeypatch.delenv("X_NUM")
    assert config.env("X_NUM", "fallback") == "fallback"


def test_the_logs_repository_is_configurable(monkeypatch):
    """It was a constant naming one person's private dataset, so a fork running
    with its own token wrote into someone else's repo."""
    monkeypatch.setenv("LOGS_DATASET", "someone/their-own-logs")
    assert config.Settings().logs_repo == "someone/their-own-logs"
    monkeypatch.delenv("LOGS_DATASET")
    assert config.Settings().logs_repo is None


def test_law_numbers_are_stripped_from_web_results():
    stripped = config.LAW_NUMBER.sub(config.LAW_PLACEHOLDER,
                                     "Under Directive (EU) 2020/2184 and Real Decreto 3/2023.")
    assert "2020/2184" not in stripped and "3/2023" not in stripped
    assert stripped.count(config.LAW_PLACEHOLDER) == 2


def test_settings_summary_does_not_leak_keys(monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "secret-value")
    summary = config.Settings().summary()
    assert summary["mistral"] is True
    assert "secret-value" not in str(summary)


# --- rendering ---------------------------------------------------------------

MESSAGES = [
    {"role": "user", "content": "What are the risks of urban wetlands in Valencia?"},
    {"role": "assistant", "route": "both",
     "content": '<p>Wetlands can create <span class="highlight">standing water</span>.</p>',
     "sources": [{"type": "evidence", "title": "4.1.9 Trade-off", "origin": "IDAlert 4.1"},
                 {"type": "ecdc", "title": "Surveillance: West Nile", "origin": "ECDC · Spain"}]},
    {"role": "user", "content": "And the co-benefits?"},
    {"role": "assistant", "route": "domain", "content": "<p>Heat management.</p>",
     "sources": [{"type": "evidence", "title": "4.1.9 Trade-off", "origin": "IDAlert 4.1"}]},
]


def test_questions_come_from_the_messages_not_a_model():
    html = render.questions_html(MESSAGES)
    assert "urban wetlands in Valencia" in html and "co-benefits" in html
    assert html.count("<li>") == 2


def test_sources_are_deduplicated_and_grouped():
    html = render.sources_html(MESSAGES)
    assert html.count("4.1.9 Trade-off") == 1
    assert "Evidence base" in html and "Surveillance data (ECDC)" in html
    assert "Web sources" not in html


def test_no_sources_says_so_rather_than_inventing_any():
    assert "No sources were retrieved" in render.sources_html([{"role": "user", "content": "x"}])


def test_markup_is_stripped_from_rendered_turns():
    html = render.turns_html(MESSAGES)
    assert "<span" not in html and "standing water" in html
    assert html.count('class="turn"') == 2


def test_markup_in_a_question_cannot_reach_the_document():
    """Tags are removed and what remains is escaped, so neither survives to be
    rendered by WeasyPrint."""
    html = render.turns_html([{"role": "user", "content": "<script>alert(1)</script>"}])
    assert "<script>" not in html and "alert(1)" in html
    assert "&amp;" in render.as_line("fish & chips")


@pytest.mark.parametrize("text,expected", [
    ("### SUMMARY\nA\n### CO-BENEFITS\nB\n### TRADE-OFFS\nC\n### UNCERTAINTIES\nD",
     ("A", "B", "C", "D")),
    ("SUMMARY\nA\nCo-Benefits\nB\nTrade offs\nC\nUncertainties and gaps\nD", ("A", "B", "C", "D")),
])
def test_section_headers_parse_in_their_variations(text, expected):
    s = render.parse_sections(text)
    assert (s.summary, s.cobenefits, s.tradeoffs, s.uncertainties) == expected


def test_unparseable_output_becomes_the_summary_rather_than_nothing():
    s = render.parse_sections("The model ignored the format entirely.")
    assert s.summary.startswith("The model ignored") and not s.cobenefits


class BriefingChat:
    def __init__(self, synthesis, audit=None, audit_fails=False):
        self.synthesis, self.audit, self.audit_fails = synthesis, audit, audit_fails
        self.calls = 0

    def __call__(self, model, messages, **kw):
        self.calls += 1
        if messages[0]["content"] is prompts.BRIEFING_AUDIT:
            if self.audit_fails:
                raise RuntimeError("timeout")
            return type("C", (), {"text": self.audit or self.synthesis, "usage": None})()
        return type("C", (), {"text": self.synthesis, "usage": None})()


DRAFT = ("### SUMMARY\nThe session looked at wetlands.\n"
         "### CO-BENEFITS\nHeat management may reduce Vibrio growth.\n"
         "### TRADE-OFFS\nStanding water may support Culex breeding. Dengue is the top risk.\n"
         "### UNCERTAINTIES\nSurveillance was not examined.")

AUDITED = ("### SUMMARY\nThe session looked at wetlands.\n"
           "### CO-BENEFITS\nHeat management may reduce Vibrio growth.\n"
           "### TRADE-OFFS\nStanding water may support Culex breeding.\n"
           "### UNCERTAINTIES\nSurveillance was not examined.")


def test_the_audit_removes_a_ranking_the_session_did_not_support():
    result = render.Briefing(BriefingChat(DRAFT, AUDITED)).build(MESSAGES)
    assert result.audited
    assert "Dengue is the top risk" in result.draft.tradeoffs
    assert "Dengue is the top risk" not in result.sections.tradeoffs


def test_a_failed_audit_keeps_the_draft_rather_than_losing_the_briefing():
    result = render.Briefing(BriefingChat(DRAFT, audit_fails=True)).build(MESSAGES)
    assert not result.audited and result.sections.summary


def test_a_session_too_short_produces_no_briefing():
    one_turn = MESSAGES[:2]
    assert render.Briefing(BriefingChat(DRAFT)).build(one_turn) is None


def test_redirects_do_not_count_towards_a_briefing():
    refusals = [{"role": "user", "content": "q"},
                {"role": "assistant", "route": "redirect", "content": "outside what I cover"}] * 3
    assert render.Briefing(BriefingChat(DRAFT)).build(refusals) is None


def test_the_briefing_html_carries_all_four_sections():
    sections = render.parse_sections(AUDITED)
    template_dir = Path(ROOT / "backend" / "templates")
    if not (template_dir / "briefing.html").exists():
        pytest.skip("templates not in this working tree")
    html = render.briefing_html(sections, MESSAGES, template_dir=template_dir)
    for text in ["The session looked at wetlands", "Heat management", "Culex breeding",
                 "Surveillance was not examined"]:
        assert text in html
