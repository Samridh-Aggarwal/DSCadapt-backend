"""The output rules that do not need a model.

Half the generation prompt was mechanical: no markdown, no banned words, no
opening pleasantry, no section numbers, no inline citations, a word count. A
model obeys those most of the time. A regex obeys them always.

Two kinds of thing live here, and the difference matters.

clean() fixes what is unambiguous. Markdown syntax is either present or not,
and removing it cannot change meaning.

check() reports what needs judgement, and never edits. "Robust" is banned when
it means good and fine when it means sturdy; a directive number is a
fabrication when it came from the model and a fact when it came from the
evidence base. Code cannot tell those apart, so it flags them and the
constraint suite decides. Silently editing a sentence whose meaning you cannot
read is worse than leaving it and saying so.
"""

import re
from dataclasses import dataclass

# Banned in the prompt because they read as machine-written. Kept as checks
# rather than substitutions: each has a legitimate sense the regex cannot see.
STYLE_WORDS = [
    "delve", "leverage", "synergy", "holistic", "paradigm", "deep dive",
    "it's important to note", "it is important to note", "it's worth mentioning",
    "I hope this helps", "let me break this down", "in conclusion",
]
AMBIGUOUS_WORDS = ["robust", "landscape", "unpack"]

# Phrases that expose the retrieval machinery to a reader who does not know it
# exists.
PIPELINE_LEAKS = [
    "the context mentions", "the context states", "from the passages",
    "the retrieved evidence", "the provided context", "based on the context",
    "the retrieved passages", "in the context block", "the retrieved documents",
]

# Recommending or ranking is the one thing the tool must never do, so it is
# checked lexically as well as by rubric.
RECOMMENDATION = [
    r"\byou should\b", r"\bwe recommend\b", r"\bI recommend\b", r"\bI'd recommend\b",
    r"\bthe best option\b", r"\bthe better option\b", r"\bmost important(?:ly)?\b",
    r"\bmore important than\b", r"\bmore severe than\b", r"\bmore significant than\b",
    r"\bprioriti[sz]e\b", r"\bthe preferred\b", r"\bwe advise\b",
]

# Categorical causal claims about what a measure does. The prompt requires
# "may increase" over "increases". Advisory only — some of these are legitimate
# statements of established fact.
CATEGORICAL = [
    r"\bwill (?:increase|reduce|cause|create|lead to|result in)\b",
    r"\b(?:increases|reduces|causes|creates|eliminates|prevents) the risk\b",
    r"\bguarantees?\b", r"\bensures that\b",
]

FILLER_OPENERS = re.compile(
    r"^\s*(?:Great question[!.]?|Certainly[!.]?|Absolutely[!.]?|Of course[!.]?|"
    r"That's an interesting (?:question|point)[!.]?|I'd be happy to help[!.]?|"
    r"Sure[,!.]?|Thanks for (?:the|your) question[!.]?)\s*", re.IGNORECASE)

# Bracketed pathway identifiers: [4.1.9], [3.1.3.1]
SECTION_TAG = re.compile(r"\[\s*\d+\.\d+(?:\.\d+)+\s*\]")

# (Author, 2024) and (Author et al., 2024). Requires the comma so that
# (2018-2024) and (EU) 2020/2184 are left alone.
INLINE_CITATION = re.compile(r"\([A-Z][A-Za-z.\-']+(?:\s+(?:et\s+al\.?|&|and)\s+[A-Z][A-Za-z.\-']+)?"
                             r"(?:\s+et\s+al\.?)?,\s*(?:19|20)\d{2}[a-z]?\)")

LAW_NUMBER = re.compile(
    r"\((?:EU|EC|EEC)\)\s*(?:No\.?\s*)?\d{1,4}/\d{2,4}"
    r"|(?:Directive|Regulation)\s+(?:\(EU\)\s*)?\d{1,4}/\d{2,4}"
    r"|(?:Royal Decree(?:-Law)?|Real Decreto|Decreto|Decree)\s+\d+/\d{4}",
    re.IGNORECASE)

REFERENCES_SPLIT = re.compile(r"(?i)\n\s*references?\s*\n")

LENGTH_BANDS = {"Brief": (0, 220), "Standard": (140, 480), "Detailed": (320, 10_000)}

# English prose runs at roughly a quarter to a third function words. Another
# European language runs at nearly zero against this list. The check is the
# ratio rather than the presence of any particular word, because a short
# answer can legitimately use only one or two of them.
#
# It matters because a non-English answer breaks the translation step: DeepL is
# asked to render French into French and passes it through, so the reader gets
# an answer that looks fine and the pipeline has no idea anything went wrong.
ENGLISH_FUNCTION_WORDS = {
    "the", "and", "of", "to", "is", "in", "that", "for", "it", "with", "as",
    "on", "be", "are", "this", "can", "may", "which", "or", "not", "by",
    "from", "at", "an", "a", "has", "have", "been", "would", "could", "these",
}
MIN_ENGLISH_RATIO = 0.08


@dataclass
class Violation:
    rule: str
    detail: str
    advisory: bool = False   # needs a human or a rubric to confirm

    def __str__(self):
        return f"{'?' if self.advisory else '!'} {self.rule}: {self.detail}"


# --- the unambiguous fixes ---------------------------------------------------

def strip_markdown(text):
    """Remove markdown syntax, keeping the words.

    app.py did this in two places with different rules, and one of them was
    text.replace("*", "") — which also removed asterisks that were doing
    something, and left the syntax it had not thought of.
    """
    text = re.sub(r"(?m)^[ \t]{0,3}#{1,6}[ \t]*", "", text)          # ## heading
    text = re.sub(r"\*\*\*([^*\n]+?)\*\*\*", r"\1", text)            # ***both***
    text = re.sub(r"\*\*([^*\n]+?)\*\*", r"\1", text)                # **bold**
    text = re.sub(r"__([^_\n]+?)__", r"\1", text)                    # __bold__
    text = re.sub(r"(?<!\w)\*([^*\n]+?)\*(?!\w)", r"\1", text)       # *italic*
    text = re.sub(r"(?<!\w)_([^_\n]+?)_(?!\w)", r"\1", text)         # _italic_
    text = re.sub(r"`([^`\n]+?)`", r"\1", text)                      # `code`
    text = re.sub(r"\[([^\]]+?)\]\(([^)\s]+?)\)", r"\1 (\2)", text)  # [text](url)
    text = re.sub(r"(?m)^[ \t]*([-*_])(?:[ \t]*\1){2,}[ \t]*$", "", text)  # --- rule
    text = re.sub(r"(?m)^[ \t]*>[ \t]?", "", text)                   # > quote
    # An asterisk touching a word on one side only is leftover syntax. One with
    # space on both sides is arithmetic or a footnote marker, and stays — the
    # original removed every asterisk in the answer, including those.
    text = re.sub(r"(?<=\w)\*(?!\w)|(?<!\w)\*(?=\w)", "", text)
    return re.sub(r"\n{3,}", "\n\n", text)


def strip_filler(text):
    return FILLER_OPENERS.sub("", text, count=1)


def strip_section_refs(text):
    """Remove bracketed pathway identifiers.

    The reader has no way to interpret [4.1.9]. The sources panel already links
    to the document, so the number is redundant even where it is correct.
    """
    text = SECTION_TAG.sub("", text)
    text = re.sub(r"[ \t]+(?=[.,;:)])", "", text)   # space left before punctuation
    return re.sub(r"[ \t]{2,}", " ", text)          # and the gap the tag left behind


def clean(text, audience="Policymaker"):
    from prompts import SECTION_NUMBERS_ALLOWED
    if not text:
        return text
    text = strip_markdown(text)
    text = strip_filler(text)
    if audience not in SECTION_NUMBERS_ALLOWED:
        text = strip_section_refs(text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# --- the reports -------------------------------------------------------------

def split_references(text):
    parts = REFERENCES_SPLIT.split(text, maxsplit=1)
    return (parts[0], parts[1]) if len(parts) == 2 else (text, "")


def check(text, audience="Policymaker", length="Standard", had_context=True):
    """Everything the prompt asks for that can be checked but not safely fixed.

    Returns violations rather than raising. In production these are logged; in
    the constraint suite they are the assertions.
    """
    from prompts import SECTION_NUMBERS_ALLOWED
    out = []
    if not text:
        return [Violation("empty", "no answer text")]

    body, references = split_references(text)
    low = body.lower()

    for phrase in STYLE_WORDS:
        if phrase in low:
            out.append(Violation("style", f"banned phrase {phrase!r}"))
    for word in AMBIGUOUS_WORDS:
        if re.search(rf"\b{word}\b", low):
            out.append(Violation("style", f"{word!r} is banned in one of its senses",
                                 advisory=True))

    for phrase in PIPELINE_LEAKS:
        if phrase in low:
            out.append(Violation("pipeline_leak", f"exposes the machinery: {phrase!r}"))

    for pattern in RECOMMENDATION:
        m = re.search(pattern, body, re.IGNORECASE)
        if m:
            out.append(Violation("recommendation", f"{m.group()!r} in {_around(body, m)!r}"))

    for pattern in CATEGORICAL:
        m = re.search(pattern, body, re.IGNORECASE)
        if m:
            out.append(Violation("conditional_language",
                                 f"categorical {m.group()!r} in {_around(body, m)!r}",
                                 advisory=True))

    if FILLER_OPENERS.match(body):
        out.append(Violation("filler", "opens with a pleasantry"))

    if audience not in SECTION_NUMBERS_ALLOWED:
        for m in SECTION_TAG.finditer(body):
            out.append(Violation("section_number", f"{m.group()} is an internal identifier"))

    for m in INLINE_CITATION.finditer(body):
        out.append(Violation("inline_citation", f"{m.group()} belongs in the reference list"))

    for m in LAW_NUMBER.finditer(body):
        out.append(Violation("law_number",
                             f"{m.group()!r} — fabrication unless it came from the evidence base",
                             advisory=True))

    if re.search(r"(?m)^\s*(?:[-*\u2022]|\d+[.)])\s+\S", body):
        out.append(Violation("format", "contains a list", advisory=True))

    for phrase in ("I can't browse", "I cannot browse", "I don't have access to real-time",
                   "I do not have access to real-time", "as an AI"):
        if phrase.lower() in low:
            out.append(Violation("capability", f"denies a capability the system has: {phrase!r}"))

    words = len(body.split())
    if words > 40:
        tokens = re.findall(r"[a-zA-Z\u00c0-\u024f']+", low)
        if tokens:
            ratio = sum(t in ENGLISH_FUNCTION_WORDS for t in tokens) / len(tokens)
            if ratio < MIN_ENGLISH_RATIO:
                out.append(Violation(
                    "language",
                    f"only {ratio:.0%} function words; this does not look like English, "
                    f"and the translation step expects it"))

    low_bound, high_bound = LENGTH_BANDS.get(length, LENGTH_BANDS["Standard"])
    if not low_bound <= words <= high_bound:
        out.append(Violation("length", f"{words} words, {length} expects "
                                       f"{low_bound}-{high_bound}", advisory=True))

    if not had_context and references.strip():
        out.append(Violation("references",
                             "reference list produced with nothing retrieved"))

    return out


def _around(text, match, width=48):
    start = max(0, match.start() - width // 2)
    return text[start:match.end() + width // 2].replace("\n", " ").strip()


def summarise(violations):
    """One line for the log. Advisory items counted separately."""
    if not violations:
        return "clean"
    hard = [v for v in violations if not v.advisory]
    soft = [v for v in violations if v.advisory]
    parts = []
    if hard:
        parts.append(f"{len(hard)} violations: " + ", ".join(sorted({v.rule for v in hard})))
    if soft:
        parts.append(f"{len(soft)} advisory: " + ", ".join(sorted({v.rule for v in soft})))
    return " | ".join(parts)
