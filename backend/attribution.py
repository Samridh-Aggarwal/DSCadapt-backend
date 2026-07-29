"""Colour each sentence of an answer by where it came from.

Green for the evidence base, amber for the web, blue for surveillance figures.
The mechanism is unchanged: embed every sentence of the answer, embed every
sentence of every source, and take the best cosine against each pool.

That mechanism is wrong, and the note is here rather than in a commit message
because it is the thing to fix next.

Cosine measures what a sentence is about. It does not measure what a sentence
claims. "Green roofs may increase Legionella risk" and "green roofs do not
increase Legionella risk" are the same topic, the same vocabulary, the same
everything — they sit within a few hundredths of each other in embedding
space. So a sentence that reverses its source scores high and gets painted
green, and green is the colour that tells a policymaker this came from
IDAlert's evidence base.

The question being asked is entailment: does the source support this claim.
There are models that answer it, they are small enough to run on a CPU next to
this one, and swapping the scorer is the only change needed — score_pools is
the seam. Everything downstream works on verdicts, not distances.

Three tells that the current signal is weak, all visible in the constants
below. Two different thresholds are needed because web text is model-written
prose about the same subject and therefore scores high against everything. A
margin is needed on top to break ties between them. And the whole thing skips
anything under forty characters, because short sentences are where topical
similarity is least informative — which is also where the most confident
claims live.
"""

import re
from dataclasses import dataclass, field

import numpy as np

import retrieval

DOMAIN_THRESHOLD = 0.875
WEB_THRESHOLD = 0.925     # higher, because web prose reads like the answer
WEB_MARGIN = 0.05         # web must beat domain by this much to win a tie
MIN_CHARS = 40

CLASSES = {"domain": "highlight", "web": "web-highlight", "surveillance": "ecdc-highlight"}
REFERENCES_SPLIT = re.compile(r"(?i)\n\s*references?\s*\n")


@dataclass
class Attribution:
    html: str = ""
    counts: dict = field(default_factory=dict)
    scored: int = 0
    skipped_short: int = 0
    embedded: int = 0

    def summary(self):
        """One line for the log.

        Always reports the work done, including when nothing was attributed —
        an answer where no sentence cleared the threshold is the interesting
        case, and it costs the same number of embeddings as one where every
        sentence did.
        """
        attributed = (", ".join(f"{k}:{v}" for k, v in sorted(self.counts.items()))
                      if self.counts else "nothing attributed")
        return (f"{attributed}; {self.scored} scored, {self.skipped_short} too short, "
                f"{self.embedded} embeddings")


def sentences(text):
    out = []
    for paragraph in text.split("\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        out.extend(s.strip() for s in re.split(r"(?<=[.!?])\s+", paragraph) if s.strip())
    return out


def surveillance_phrases(block, answer):
    """Figures from the surveillance block that appear in the answer.

    Purely lexical, matching on the number. A correctly cited figure written as
    "just over three hundred" instead of "332" is never caught.
    """
    if not block or not answer:
        return []
    numbers = set()
    for line in block.split("\n"):
        if any(k in line.lower() for k in ("cases:", "deaths:")):
            numbers.update(m for m in re.findall(r":\s*(\d+)", line) if 5 < int(m) < 100_000)

    found = []
    for number in numbers:
        for pattern in (rf"\b{number}\s+(?:reported\s+)?cases?\b",
                        rf"\b{number}\s+(?:reported\s+)?deaths?\b",
                        rf"\b{number}\s+in\s+\d{{4}}\b"):
            match = re.search(pattern, answer, re.IGNORECASE)
            if match:
                found.append(match.group().lower())
    return list(dict.fromkeys(found))[:3]


def score_pools(claims, pools, embed):
    """Best score for each claim against each pool.

    The seam. Today this is cosine similarity, which answers "is this about the
    same thing". An entailment model answers "does this follow from that", and
    dropping one in means replacing this function and nothing else.
    """
    names = [n for n, texts in pools.items() if texts]
    if not claims or not names:
        return {n: np.zeros(len(claims)) for n in pools}, 0

    flat, spans = [], {}
    for name in names:
        spans[name] = (len(flat), len(flat) + len(pools[name]))
        flat.extend(pools[name])

    matrix = retrieval.normalise(retrieval.embed_all(embed, claims + flat))
    claim_vectors = matrix[:len(claims)]
    source_vectors = matrix[len(claims):]

    scores = {n: np.zeros(len(claims)) for n in pools}
    for name in names:
        start, end = spans[name]
        scores[name] = (claim_vectors @ source_vectors[start:end].T).max(axis=1)
    return scores, len(claims) + len(flat)


def classify(domain, web):
    """Which pool a sentence belongs to, or none."""
    domain_ok, web_ok = domain >= DOMAIN_THRESHOLD, web >= WEB_THRESHOLD
    if not domain_ok and not web_ok:
        return None
    if domain_ok and web_ok:
        return "web" if web - domain > WEB_MARGIN else "domain"
    return "domain" if domain_ok else "web"


def wrap(body, tagged):
    """Wrap each tagged sentence in its span.

    Longest first through placeholders, so a sentence that contains another
    does not end up with nested spans.
    """
    result, placeholders = body, []
    for sentence, kind in sorted(tagged, key=lambda t: len(t[0]), reverse=True):
        match = re.search(re.escape(sentence), result)
        if not match:
            continue
        token = f"{{_HL{len(placeholders)}_}}"
        placeholders.append((token, f'<span class="{CLASSES[kind]}">{match.group()}</span>'))
        result = result[:match.start()] + token + result[match.end():]
    for token, span in placeholders:
        result = result.replace(token, span)
    return result


def annotate(answer, hits=None, web_text="", surveillance_block="", embed=None):
    """Attribute an answer to its sources. English only — a translated answer
    will not match English source text, so the caller skips this entirely."""
    result = Attribution(html=answer)
    if not answer or embed is None:
        return result

    # never highlight inside the reference list
    parts = REFERENCES_SPLIT.split(answer, maxsplit=1)
    body = parts[0]
    tail = ("\n\nReferences\n" + parts[1]) if len(parts) == 2 else ""

    # Surveillance is lexical and wins outright: a sentence carrying a figure
    # from the data is attributed to the data and left out of the scoring.
    figures = surveillance_phrases(surveillance_block, body)

    tagged, scorable = [], []
    for sentence in sentences(body):
        low = sentence.lower()
        if any(phrase in low for phrase in figures):
            tagged.append((sentence, "surveillance"))
        elif len(sentence) >= MIN_CHARS:
            scorable.append(sentence)
        else:
            result.skipped_short += 1

    pools = {
        "domain": [s for hit in (hits or []) for s in sentences(hit.chunk.text)],
        "web": sentences(web_text) if web_text else [],
    }
    result.scored = len(scorable)

    if scorable and (pools["domain"] or pools["web"]):
        try:
            scores, embedded = score_pools(scorable, pools, embed)
            result.embedded = embedded
            for i, sentence in enumerate(scorable):
                kind = classify(float(scores["domain"][i]), float(scores["web"][i]))
                if kind:
                    tagged.append((sentence, kind))
        except Exception as err:
            # Attribution is decoration. Losing it must not lose the answer.
            print(f"[ATTRIBUTION] scoring failed, returning unhighlighted: {err}")
            return result

    for _, kind in tagged:
        result.counts[kind] = result.counts.get(kind, 0) + 1
    result.html = wrap(body, tagged) + tail if tagged else answer
    return result
