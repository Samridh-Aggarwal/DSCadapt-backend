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

import hashlib
import html as html_
import re
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape
from markupsafe import Markup, escape

import labels
import logos
import prompts
import translate

TEMPLATE_DIR = Path(__file__).parent / "templates"

# Printed in the closing band of every document. Need to change it here and both
# documents change; there is nowhere else it is written down (except on frontend, where I added it as a placeholder at the bottom of the landing page).
CONTACT_EMAIL = "[CONTACT EMAIL]"
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


REFERENCE_HEADING = re.compile(
    r"(?im)^[ \t]*#{0,4}[ \t]*\**[ \t]*references?[ \t]*:?[ \t]*\**[ \t]*$")

# The answer reaching this module is already HTML: pipeline.py wraps it in
# paragraphs and attribution.py marks each sentence with where it came from,
# so the colour a reader sees in the app is provenance, not decoration. The
# transcript keeps it. Anything outside this list is dropped rather than
# trusted, because part of the markup was written by the model.
ALLOWED_TAGS = {"p": set(), "br": set(), "ul": set(), "ol": set(), "li": set(),
                "strong": {"class"}, "b": set(), "em": {"class"}, "i": set(),
                "span": {"class"}}
ALLOWED_CLASSES = {"highlight", "web-highlight", "ecdc-highlight", "disease-term"}
TAG = re.compile(r"<(/?)([a-zA-Z][a-zA-Z0-9]*)([^>]*)>")
CLASS_ATTR = re.compile(r'class\s*=\s*"([^"]*)"')
IS_HTML = re.compile(r"(?i)</?(?:p|span|strong|em|br|ul|ol|li)\b")


def sanitise(markup):
    """Keep the tags and classes the answer is allowed to carry, escape the rest.

    Whitelisting rather than stripping, because the highlight spans are the
    only record of which sentence came from the evidence base, which from the
    web and which from surveillance data. Stripping them, which is what this
    module used to do, threw that away and left the reader unable to tell.
    """
    out, pos = [], 0
    for match in TAG.finditer(markup):
        out.append(str(escape(markup[pos:match.start()])))
        pos = match.end()
        closing, name, attrs = match.group(1), match.group(2).lower(), match.group(3)
        if name not in ALLOWED_TAGS:
            continue
        if closing:
            out.append(f"</{name}>")
        elif name == "br":
            out.append("<br>")
        else:
            kept = ""
            if "class" in ALLOWED_TAGS[name]:
                found = CLASS_ATTR.search(attrs)
                if found:
                    names = [c for c in found.group(1).split() if c in ALLOWED_CLASSES]
                    if names:
                        kept = ' class="' + " ".join(names) + '"'
            out.append(f"<{name}{kept}>")
    out.append(str(escape(markup[pos:])))
    return "".join(out)

def inline(text):
    """One line of answer text, however it arrived.

    Already HTML, which is the normal case, so it is whitelisted. Plain text,
    which is what the older sessions in a saved history hold, so *italics* and
    **bold** are honoured and everything else is escaped.
    """
    text = text or ""
    if IS_HTML.search(text):
        return Markup(sanitise(text))
    out = str(escape(text))
    out = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", out)
    return Markup(re.sub(r"\*([^*\n]+)\*", r"<em>\1</em>", out))


def rich(text):
    """A whole response, however it arrived.

    Already HTML, so it is whitelisted and the highlighting survives. Plain
    text, so a blank line starts a paragraph and a block whose every line
    opens with a dash becomes a list, because a block with one stray dash in
    it is prose about a range, not a list.
    """
    text = text or ""
    if IS_HTML.search(text):
        return Markup(sanitise(text))
    out = []
    for block in re.split(r"\n\s*\n", text.strip()):
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if lines and all(line.startswith(("- ", "\u2022 ")) for line in lines):
            out.append("<ul>" + "".join(f"<li>{inline(line[2:])}</li>"
                                        for line in lines) + "</ul>")
        elif lines:
            out.append(f"<p>{inline(' '.join(lines))}</p>")
    return Markup("".join(out))


def css_str(text):
    """Quote a label for a CSS content: property, which the page numbers use."""
    return Markup('"' + str(text).replace("\\", "\\\\").replace('"', '\\"') + '"')


def _env(template_dir):
    """Jinja for the PDF templates. Autoescaping on: the model writes this text."""
    env = Environment(loader=FileSystemLoader(str(template_dir)),
                      autoescape=select_autoescape(["html"]))
    env.filters["inline"] = inline
    env.filters["rich"] = rich
    env.filters["css_str"] = css_str
    return env

def split_references(text):
    """Separate a response from the reference list it closes with.

    The transcript sets the two differently, so they cannot stay in one blob.
    By the time an answer reaches here the list is a final paragraph reading
    References followed by one line break per entry, so the split is on that
    paragraph rather than on the blank line attribution.py looks for upstream.
    Plain text is handled too, for a history saved before any of this.
    """
    text = text or ""
    if re.search(r"(?i)<p[\s>]", text):
        blocks = re.split(r"(?i)(?=<p[\s>])", text)
        for i, block in enumerate(blocks):
            if not re.match(r"(?i)\s*references?\b",
                            re.sub(r"<[^>]+>", " ", block).strip()):
                continue
            inner = re.sub(r"(?is)^\s*<p[^>]*>\s*(?:<[^>]+>\s*)*references?\s*:?\s*",
                           "", block)
            inner = re.sub(r"(?is)</p>\s*$", "", inner)
            items = [re.sub(r"^\s*(?:[-\u2022*]|\d{1,2}[.)])\s*", "", piece).strip()
                     for piece in re.split(r"(?i)<br\s*/?>", inner)]
            return "".join(blocks[:i]).strip(), [item for item in items if item]
        return text.strip(), []
    match = REFERENCE_HEADING.search(text)
    if not match:
        return text.strip(), []
    body = text[:match.start()].strip()
    items = [re.sub(r"^\s*(?:[-\u2022*]|\d{1,2}[.)])\s+", "", line).strip()
             for line in text[match.end():].splitlines() if line.strip()]
    return body, [item for item in items if item]


def exchanges(history):
    """Pair every question with the response that followed it.

    An assistant message with nothing before it still gets an entry rather
    than being dropped, with an empty question; the template leaves the
    question box out when there is nothing to put in it.
    """
    rows, i, size = [], 0, len(history)
    while i < size:
        role = history[i].get("role", "")
        content = history[i].get("content", "") or ""
        if role == "user":
            question = re.sub(r"\s+", " ", content).strip()
            response = ""
            if i + 1 < size and history[i + 1].get("role") == "assistant":
                response = history[i + 1].get("content", "") or ""
                i += 2
            else:
                i += 1
        elif role == "assistant":
            question, response = "", content
            i += 1
        else:
            i += 1
            continue
        body, references = split_references(response)
        if question or body:
            rows.append({"question": question, "response": body,
                         "references": references})
    return rows


def session_ref(history):
    """A short stable name for one conversation.

    The same conversation exported twice gets the same reference, which is
    the point: it is what someone quotes when they come back about a document
    weeks later. The request carries no session id to use instead.
    """
    joined = "\n".join(f"{m.get('role', '')}:{m.get('content', '')}"
                       for m in history)
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()[:8].upper()


def _lang_code(language):
    """The html lang attribute. DeepL's codes serve once lowercased."""
    code = translate.LANGUAGES.get(language)
    return code.split("-")[0].lower() if code else "en"


def transcript_html(history, audience="Policymaker", country="All Europe",
                    language="English", length="Standard",
                    template_dir=TEMPLATE_DIR):
    rows = exchanges(history)
    if not rows:
        return None
    now = datetime.now()
    return _env(template_dir).get_template("transcript.html").render(
        lang=_lang_code(language),
        date=f"{now.day} {now:%B %Y}",
        session_ref=session_ref(history),
        contact=CONTACT_EMAIL,
        logos=logos.LOGOS,
        settings={"country": country, "mode": audience,
                  "language": language, "length": length},
        exchanges=rows,
        L=labels.transcript(language),
    )


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
                      language="English", length="Standard",
                      template_dir=TEMPLATE_DIR):
    html = transcript_html(history, audience, country, language, length,
                           template_dir)
    return _write_pdf(html, "idalert_transcript", template_dir) if html else None

def render_briefing(sections, messages, audience="Policymaker", country="All Europe",
                    language="English", template_dir=TEMPLATE_DIR):
    html = briefing_html(sections, messages, audience, country, language, template_dir)
    return _write_pdf(html, "idalert_briefing", template_dir)
