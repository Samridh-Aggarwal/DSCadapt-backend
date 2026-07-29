"""Review a policy document for infectious-disease risks it does not address.

The shape of this is the strongest idea in the system and it stays exactly as
it was. A paragraph of the uploaded document matches a pathway document; the
manifest says which adaptation measure that document is about; the ontology
says which trade-off mechanisms that measure has and which diseases those
reach; surveillance says which of those diseases matter in this country. The
model is handed that list and may name nothing outside it, and a post-filter
throws away any comment that does.

That is ontology-constrained generation with a validating filter — the thing
GraphRAG approximates by extracting a graph with an LLM, except the graph here
was written by domain experts, so it is worth more and costs nothing to walk.

Three things fixed:

  - every paragraph of the document went to the embedding API in a single call.
    A sixty-thousand-character document splits into two hundred-odd paragraphs,
    and that request fails.

  - matching was a Python loop over every document paragraph crossed with every
    corpus chunk, computing cosine one pair at a time. It is one matrix
    multiply.

  - the whole uploaded document was written into the log, which is pushed to a
    dataset. Someone reviewing an unpublished national strategy put it there.
    Only a length and a hash are recorded now.
"""

import hashlib
import re
from dataclasses import dataclass, field

import numpy as np

import ecdc
import ontology
import prompts
import retrieval

MAX_CHARS = 60_000
SCOPE_CHARS = 6_000
MIN_PARAGRAPH = 40
EVIDENCE_PER_MEASURE = 3
EVIDENCE_CHARS = 2500
SCOPE_MODEL = "ministral-8b-latest"
REVIEW_MODEL = "mistral-small-latest"

# What "relevant to this measure" means when picking which of its chunks to
# show the model. A fixed probe rather than the document's own words, so the
# evidence chosen is about disease risk rather than about whatever vocabulary
# the document happens to use.
PROBE = ("infectious disease risks of this measure: pathogens, mosquito and tick vectors, "
         "vector-borne and waterborne transmission, standing water, contamination, "
         "health trade-offs")

OUT_OF_SCOPE = (
    "This does not look like a climate adaptation, environmental, or health policy document, "
    "so there is nothing here for the reviewer to assess. It checks adaptation strategies, "
    "plans and briefs for overlooked infectious-disease trade-offs. Paste a document in that "
    "area and it will review it.")

NO_MATCH = "No passage in this document matched the evidence base closely enough to review."

NO_MEASURE = ("Nothing in this document maps to a measure with an infectious-disease trade-off "
              "in the evidence base for this location.")

NOTE = ("Selective review, {n} consideration{s} found. Comments appear only where the evidence "
        "base links a measure in your document to an infectious-disease risk the document does "
        "not address. No comment is not approval, and this review does not check whether your "
        "document's facts are correct.")


@dataclass
class Review:
    note: str = ""
    comments: str = "NO GROUNDED CONSIDERATIONS"
    shortlist: list = field(default_factory=list)
    measures: list = field(default_factory=list)
    allowed: list = field(default_factory=list)
    draft: str = ""
    dropped: list = field(default_factory=list)
    out_of_scope: bool = False
    scope_checked: bool = True
    n_comments: int = 0

    def log_record(self, country, text, include_comments=True):
        """What goes in the log. Deliberately not the document.

        The reviewer is used on drafts that are not public. Storing the text in
        a hosted dataset because it happened to pass through here is not a
        trade anyone agreed to.

        The comments are a separate decision. They quote short passages, so
        they carry fragments of the document — but without them the reviewer
        cannot be evaluated at all, and there is no seeing what the allowlist
        filter threw away. Same switch as the question text on /ask.
        """
        record = {
            "country": country,
            "document_chars": len(text),
            "document_sha1": hashlib.sha1(text.encode("utf-8")).hexdigest()[:16],
            "shortlist": self.shortlist,
            "measures": self.measures,
            "allowed_diseases": self.allowed,
            "n_comments": self.n_comments,
            "dropped_comments": len(self.dropped),
            "out_of_scope": self.out_of_scope,
            "scope_checked": self.scope_checked,
        }
        if include_comments:
            record["draft"] = self.draft
            record["comments"] = self.comments
            record["dropped"] = self.dropped
        return record


# --- comment handling --------------------------------------------------------

def split_comments(text):
    """Keep well-formed PASSAGE/CONSIDERATION pairs, discard any prose around them."""
    if not text or "NO GROUNDED CONSIDERATIONS" in text.upper():
        return []
    return [block.strip() for block in re.split(r"\n\s*\n", text)
            if "PASSAGE:" in block.upper() and "CONSIDERATION:" in block.upper()]


def surface_forms(disease):
    """Every way a disease might be written, for the allowlist filter."""
    forms = {disease}
    record = ecdc.DISEASES.get(disease, {})
    if record.get("ecdc"):
        forms.add(record["ecdc"])
    forms.update(record.get("aliases", []))
    bracket = re.search(r"\(([^)]+)\)", disease)
    if bracket:
        forms.add(bracket.group(1))
        forms.add(disease[:bracket.start()].strip())
    return {f.lower() for f in forms if f}


def names_allowed(comment, allowed):
    """A comment survives only if it names a disease from the filtered set.

    The geography guard. Without it the model can name a disease that is not
    established in the user's country, which for a reviewer is worse than
    saying nothing.
    """
    low = comment.lower()
    return any(form in low for disease in allowed for form in surface_forms(disease))


# --- the reviewer ------------------------------------------------------------

class Reviewer:
    def __init__(self, index, surveillance, measures, embed, chat):
        self.index = index
        self.surveillance = surveillance
        self.measures = measures          # source filename -> measure name
        self.embed = embed
        self.chat = chat
        self._probe = None

    def probe_vector(self):
        if self._probe is None:
            vec = np.asarray(self.embed([PROBE])[0], dtype=float)
            self._probe = vec / (np.linalg.norm(vec) or 1.0)
        return self._probe

    # -- scope ---------------------------------------------------------------

    def in_scope(self, text):
        """Meaning-based, so a paper that merely shares vocabulary is rejected.

        Fails open: a transient API problem should not block a real review.
        The caller records whether the check actually ran.
        """
        try:
            completion = self.chat(
                model=SCOPE_MODEL,
                messages=[{"role": "system", "content": prompts.REVIEW_SCOPE},
                          {"role": "user", "content": text[:SCOPE_CHARS]}],
                temperature=0, max_tokens=5)
            return not completion.text.strip().lower().startswith("out"), True
        except Exception as err:
            print(f"[REVIEW] scope check failed, accepting the document: {err}")
            return True, False

    # -- matching ------------------------------------------------------------

    def match_documents(self, text, top_per_chunk, min_score):
        """Which corpus documents this text resembles, and how strongly.

        Scores accumulate per source, so a document matching one pathway in
        several places outranks one matching another pathway once. That is the
        original behaviour and it is arguable — many weak matches can outweigh
        a single strong one — but changing it changes which measures get
        reviewed, so it waits for the eval set.
        """
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text)
                      if len(p.strip()) > MIN_PARAGRAPH] or [text]
        vectors = retrieval.normalise(retrieval.embed_all(self.embed, paragraphs))
        similarity = vectors @ self.index.vectors.T          # paragraphs x chunks

        scores = {}
        for row in similarity:
            order = np.argsort(row)[::-1]
            if row[order[0]] < min_score:
                continue
            for idx in order[:top_per_chunk]:
                source = self.index.chunks[idx].source
                # Only documents that map to a measure can ever produce a
                # comment. Counting the glossary and the case studies filled
                # shortlist slots with sources the graph walk then discarded.
                if source not in self.measures:
                    continue
                scores[source] = scores.get(source, 0.0) + float(row[idx])
        return scores

    def evidence_for(self, source):
        """The chunks from this document that are most about disease risk."""
        indices = [i for i, c in enumerate(self.index.chunks) if c.source == source]
        if not indices:
            return ""
        probe = self.probe_vector()
        indices.sort(key=lambda i: float(self.index.vectors[i] @ probe), reverse=True)
        return " ".join(self.index.chunks[i].text
                        for i in indices[:EVIDENCE_PER_MEASURE])[:EVIDENCE_CHARS]

    def measure_blocks(self, shortlist, country):
        """One block per measure: its mechanisms, its permitted diseases, its evidence."""
        blocks, seen, allowed = [], [], set()
        for source in shortlist:
            measure = self.measures.get(source)
            if not measure or measure in seen or measure not in ontology.MEASURE_RISKS:
                continue

            pairs = ontology.MEASURE_RISKS[measure]
            candidates = sorted({d for _, category in pairs
                                 for d in ontology.CATEGORY_DISEASES[category]
                                 if self.surveillance.relevant(d, country)})
            if not candidates:
                continue

            seen.append(measure)
            allowed |= set(candidates)
            mechanisms = sorted({m for m, _ in pairs})
            blocks.append(
                f"MEASURE: {measure}\n"
                f"HOW IT CAN RAISE RISK: {'; '.join(mechanisms)}\n"
                f"DISEASES TO CONSIDER (already filtered for {country}): {', '.join(candidates)}\n"
                f"EVIDENCE: {self.evidence_for(source)}")
        return blocks, seen, sorted(allowed)

    # -- the whole thing -----------------------------------------------------

    def review(self, text, country="All Europe", top_per_chunk=3, shortlist_n=6,
               min_score=0.70):
        if not text or not text.strip():
            raise ValueError("no document text provided")
        if len(text) > MAX_CHARS:
            raise ValueError(
                f"This document is too long for the reviewer ({len(text):,} characters). "
                f"It is built for briefs and strategies up to about {MAX_CHARS:,} characters. "
                f"Paste the relevant section instead.")

        result = Review()
        ok, checked = self.in_scope(text)
        result.scope_checked = checked
        if not ok:
            result.out_of_scope = True
            result.note = OUT_OF_SCOPE
            return result

        scores = self.match_documents(text, top_per_chunk, min_score)
        if not scores:
            result.note = NO_MATCH
            return result

        ranked = sorted(scores.items(), key=lambda kv: -kv[1])[:shortlist_n]
        result.shortlist = [s.replace(".docx.txt", "").replace(".txt", "") for s, _ in ranked]

        blocks, measures, allowed = self.measure_blocks([s for s, _ in ranked], country)
        result.measures, result.allowed = measures, allowed
        if not blocks:
            result.note = NO_MEASURE
            return result

        message = f"DOCUMENT:\n\n{text}\n\n---\n\nMEASURES:\n\n" + "\n\n".join(blocks)
        completion = self.chat(
            model=REVIEW_MODEL,
            messages=[{"role": "system", "content": prompts.REVIEW},
                      {"role": "user", "content": message}],
            temperature=0, random_seed=7, max_tokens=4000, reasoning_effort="high")

        drafted = split_comments(completion.text)
        kept = [c for c in drafted if names_allowed(c, allowed)]
        result.draft = "\n\n".join(drafted)
        result.dropped = [c for c in drafted if c not in kept]
        result.comments = "\n\n".join(kept) if kept else "NO GROUNDED CONSIDERATIONS"
        result.n_comments = len(kept)
        result.note = NOTE.format(n=len(kept), s="" if len(kept) == 1 else "s")
        return result
