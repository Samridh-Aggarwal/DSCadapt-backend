"""Two documents: a transcript of a session, and a briefing derived from one.

The transcript is mechanical — turns in, turns out. The briefing is not: a
model consolidates the session into four sections, then a second pass audits
that draft against the session and deletes anything the session does not
support. The audit only ever removes; it is not allowed to add or rephrase.

The parts that must not be invented are built in Python and never by a model.
The list of questions comes from the message objects. So does the list of
sources. A briefing that fabricated either would be worse than no briefing,
and the only way to be sure it cannot is not to ask.
"""

import html as html_
import re
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import prompts

TEMPLATE_DIR = Path(__file__).parent / "templates"
SUBSTANTIVE_ROUTES = ("domain", "both", "web")
MIN_TURNS_FOR_BRIEFING = 2

SECTION_HEADER = re.compile(
    r"(?im)^[ \t]*#{0,4}[ \t]*(SUMMARY|CO[ \t\-]?BENEFITS?|TRADE[ \t\-]?OFFS?|"
    r"UNCERTAINT(?:Y|IES)(?:[ \t]+AND[ \t]+GAPS)?)\b[^\n]*\n")

SOURCE_GROUPS = [("evidence", "Evidence base and IDAlert publications"),
                 ("ecdc", "Surveillance data (ECDC)"),
                 ("web", "Web sources")]


@dataclass
class Sections:
    summary: str = ""
    cobenefits: str = ""
    tradeoffs: str = ""
    uncertainties: str = ""

    def empty(self):
        return not any((self.summary, self.cobenefits, self.tradeoffs, self.uncertainties))

    def as_text(self):
        return (f"### SUMMARY\n{self.summary}\n\n"
                f"### CO-BENEFITS\n{self.cobenefits}\n\n"
                f"### TRADE-OFFS\n{self.tradeoffs}\n\n"
                f"### UNCERTAINTIES\n{self.uncertainties}")


# --- escaping ----------------------------------------------------------------

def as_line(content):
    return html_.escape(re.sub(r"<[^>]+>", "", content)).strip().replace("\n", "<br>")


def as_paragraphs(content):
    content = re.sub(r"(?i)</p>\s*<p[^>]*>", "\n\n", content)
    content = re.sub(r"(?i)<br\s*/?>", "\n", content)
    content = re.sub(r"(?i)</?p[^>]*>", "\n\n", content)
    content = html_.escape(re.sub(r"<[^>]+>", "", content))
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", content) if p.strip()]
    return "".join(f"<p>{p}</p>" for p in
                   (p.replace("\n", "<br>") for p in paragraphs))


# --- deterministic sections --------------------------------------------------

def meta_line(country, audience, language):
    fields = [f"Generated {datetime.now().strftime('%d %B %Y')}", f"Country: {country}",
              f"Mode: {audience}", f"Language: {language}"]
    return '<span class="sep">|</span>'.join(
        f"<span>{html_.escape(f)}</span>" for f in fields)


def questions_html(messages):
    asked = [as_line(m.get("content", "")) for m in messages
             if m.get("role") == "user" and m.get("content", "").strip()]
    if not asked:
        return "<p>No questions recorded.</p>"
    return "<ul class='q-list'>" + "".join(f"<li>{q}</li>" for q in asked) + "</ul>"


def sources_html(messages):
    """Built from the message objects, so it cannot contain a source that was
    never retrieved."""
    seen, groups = set(), {kind: [] for kind, _ in SOURCE_GROUPS}
    for message in messages:
        for source in message.get("sources") or []:
            key = (source.get("title", ""), source.get("origin", ""))
            if key in seen:
                continue
            seen.add(key)
            groups.setdefault(source.get("type", "evidence"), []).append(source)

    out = []
    for kind, label in SOURCE_GROUPS:
        if not groups.get(kind):
            continue
        out.append(f"<div class='src-group'><div class='src-label'>{label}</div>")
        for source in groups[kind]:
            out.append(f"<div class='src-item'>{html_.escape(source.get('title',''))}"
                       f"<span class='src-origin'> — "
                       f"{html_.escape(source.get('origin',''))}</span></div>")
        out.append("</div>")
    return "".join(out) or "<p>No sources were retrieved during this session.</p>"


def turns_html(history):
    blocks, i, n = [], 0, len(history)
    while i < n:
        role = history[i].get("role", "")
        content = history[i].get("content", "")
        if role == "user":
            question, answer = as_line(content), ""
            if i + 1 < n and history[i + 1].get("role") == "assistant":
                answer = as_paragraphs(history[i + 1].get("content", ""))
                i += 2
            else:
                i += 1
            block = ""
            if question:
                block += f'<div class="label q">Query</div><div class="q-text">{question}</div>'
            if answer:
                block += ('<div class="label a" style="margin-top:5mm">Response</div>'
                          f'<div class="a-text">{answer}</div>')
            if block:
                blocks.append(f'<div class="turn">{block}</div>')
        elif role == "assistant":
            answer = as_paragraphs(content)
            if answer:
                blocks.append('<div class="turn"><div class="label a">Response</div>'
                              f'<div class="a-text">{answer}</div></div>')
            i += 1
        else:
            i += 1
    return "".join(blocks)


def session_text(messages):
    lines = []
    for message in messages:
        content = re.sub(r"<[^>]+>", "", message.get("content", "")).strip()
        if not content:
            continue
        if message.get("role") == "user":
            lines.append("USER QUESTION: " + content)
        elif message.get("role") == "assistant":
            lines.append("TOOL RESPONSE: " + content)
    return "\n\n".join(lines)


# --- briefing synthesis ------------------------------------------------------

def parse_sections(text):
    text = text.replace("**", "").replace("*", "")
    matches = list(SECTION_HEADER.finditer(text))
    if not matches:
        return Sections(summary=text.strip())

    out = Sections()
    for i, match in enumerate(matches):
        key = re.sub(r"[ \t\-]", "", match.group(1).upper())
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[match.end():end].strip()
        if key.startswith("SUMMARY"):
            out.summary = body
        elif key.startswith("COBENEFIT"):
            out.cobenefits = body
        elif key.startswith("TRADEOFF"):
            out.tradeoffs = body
        elif key.startswith("UNCERTAINT"):
            out.uncertainties = body
    return out if not out.empty() else Sections(summary=text.strip())


@dataclass
class BriefingResult:
    sections: Sections = field(default_factory=Sections)
    draft: Sections = field(default_factory=Sections)
    audited: bool = False
    turns: int = 0


class Briefing:
    """Consolidate, then audit. The audit is the point."""

    def __init__(self, chat, model="mistral-small-latest"):
        self.chat = chat
        self.model = model

    def synthesise(self, session):
        completion = self.chat(
            model=self.model,
            messages=[{"role": "system", "content": prompts.BRIEFING_SYNTHESIS},
                      {"role": "user",
                       "content": f"Here is the full session to consolidate into the "
                                  f"briefing.\n\n{session}"}],
            temperature=0.2, max_tokens=2000)
        return parse_sections(completion.text)

    def audit(self, sections, session):
        """Second pass. Removes what the session does not support, nothing else.

        Returns the draft unchanged if it fails: an unaudited briefing is worse
        than an audited one, and no briefing at all is worse than both.
        """
        try:
            completion = self.chat(
                model=self.model,
                messages=[{"role": "system", "content": prompts.BRIEFING_AUDIT},
                          {"role": "user",
                           "content": f"SESSION:\n\n{session}\n\n---\n\nDRAFT BRIEFING:"
                                      f"\n\n{sections.as_text()}"}],
                temperature=0.1, max_tokens=2000)
        except Exception as err:
            print(f"[BRIEFING] audit failed, keeping the unaudited draft: {err}")
            return sections, False
        audited = parse_sections(completion.text)
        return (audited, True) if not audited.empty() else (sections, False)

    def build(self, messages):
        substantive = [m for m in messages
                       if m.get("role") == "assistant" and m.get("route") in SUBSTANTIVE_ROUTES]
        if len(substantive) < MIN_TURNS_FOR_BRIEFING:
            return None

        session = session_text(messages)
        draft = self.synthesise(session)
        sections, audited = self.audit(draft, session)
        return BriefingResult(sections=sections, draft=draft, audited=audited,
                              turns=len(substantive))


# --- PDF ---------------------------------------------------------------------

def _write_pdf(html, stem, template_dir):
    import weasyprint
    path = Path(tempfile.gettempdir()) / f"{stem}_{datetime.now():%Y%m%d_%H%M%S}.pdf"
    weasyprint.HTML(string=html, base_url=str(template_dir)).write_pdf(str(path))
    return str(path)


def transcript_html(history, audience="Policymaker", country="All Europe",
                    language="English", template_dir=TEMPLATE_DIR):
    turns = turns_html(history)
    if not turns:
        return None
    template = (Path(template_dir) / "transcript.html").read_text(encoding="utf-8")
    return (template
            .replace("{{META}}", meta_line(country, audience, language))
            .replace("{{TURNS}}", turns))


def briefing_html(sections, messages, audience="Policymaker", country="All Europe",
                  language="English", template_dir=TEMPLATE_DIR):
    template = (Path(template_dir) / "briefing.html").read_text(encoding="utf-8")
    return (template
            .replace("{{META}}", meta_line(country, audience, language))
            .replace("{{QUESTIONS}}", questions_html(messages))
            .replace("{{SUMMARY}}", as_paragraphs(sections.summary))
            .replace("{{COBENEFITS}}", as_paragraphs(sections.cobenefits))
            .replace("{{TRADEOFFS}}", as_paragraphs(sections.tradeoffs))
            .replace("{{UNCERTAINTIES}}", as_paragraphs(sections.uncertainties))
            .replace("{{SOURCES}}", sources_html(messages)))


def render_transcript(history, audience="Policymaker", country="All Europe",
                      language="English", template_dir=TEMPLATE_DIR):
    html = transcript_html(history, audience, country, language, template_dir)
    return _write_pdf(html, "idalert_transcript", template_dir) if html else None


def render_briefing(sections, messages, audience="Policymaker", country="All Europe",
                    language="English", template_dir=TEMPLATE_DIR):
    html = briefing_html(sections, messages, audience, country, language, template_dir)
    return _write_pdf(html, "idalert_briefing", template_dir)
