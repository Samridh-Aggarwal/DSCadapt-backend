"""Search the corpus.

Same scoring as before — cosine plus a rare-term keyword signal, blended
0.85/0.15, deduplicated to one chunk per section — but computed as matrix
operations instead of a loop per chunk.

The old version, per query: lowercased all 309 chunk bodies once for every
query word to count document frequency, lowercased them all again while
scoring, called cosine_sim 309 times each doing its own norm, then recomputed
cosine a fourth time during deduplication. Roughly four million character
operations to rank a corpus that fits in memory twice over.

Two other things changed:

  - the query vector is returned rather than hidden. app.py only produced one
    inside retrieve(), so a web-routed query had no vector, and the publication
    and scoping-review matchers were skipped along with everything else.

  - a hit carries its cosine and keyword scores separately. Only the blended
    number was ever logged, which made the 0.85/0.15 split impossible to tune
    from production traffic.

Reciprocal rank fusion is implemented but not the default. It is the right
answer to blending two scores that do not share a scale, and it is what should
be used once there is an eval set to confirm it — switching blind would change
every ranking with no way to tell whether it helped.
"""

import re
from dataclasses import dataclass

import numpy as np

DENSE_WEIGHT = 0.85
CANDIDATES = 20        # scored, then deduplicated by section
RRF_K = 60             # Cormack et al., SIGIR 2009
EMBED_BATCH = 100

STOPWORDS = {
    "what", "that", "this", "from", "with", "have", "will", "been",
    "were", "they", "their", "there", "which", "would", "could",
    "should", "about", "these", "those", "being", "does", "into",
    "than", "them", "then", "when", "where", "your", "also", "some",
    "more", "other", "such", "each", "most", "very", "here", "after",
}


def embed_all(embed_fn, texts, batch=EMBED_BATCH):
    """Embed in batches.

    Everything that embeds goes through here. /evaluate used to call the API
    once with every paragraph of an uploaded document, which fails on anything
    longer than a short brief.
    """
    if not texts:
        return np.zeros((0, 0))
    vectors = []
    for i in range(0, len(texts), batch):
        vectors.extend(embed_fn(texts[i:i + batch]))
    return np.asarray(vectors, dtype=float)


def normalise(matrix):
    """Unit rows, so a dot product is a cosine."""
    if matrix.size == 0:
        return matrix
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def query_terms(query):
    return {w.lower() for w in re.findall(r"\b\w+\b", query)
            if len(w) > 2 and w.lower() not in STOPWORDS}


@dataclass
class Hit:
    chunk: object
    score: float          # the blended score the ranking used
    cosine: float
    keyword: float
    index: int


class Index:
    """Chunks, their vectors, and a lowercase copy of each body.

    The lowercase copies are the whole trick for the keyword half: they are
    built once instead of on every query word of every request.
    """

    def __init__(self, chunks, vectors):
        if len(chunks) != len(vectors):
            raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors")
        self.chunks = chunks
        self.vectors = normalise(np.asarray(vectors, dtype=float))
        self._lower = [c.text.lower() for c in chunks]

    @classmethod
    def build(cls, chunks, embed_fn):
        return cls(chunks, embed_all(embed_fn, [c.text for c in chunks]))

    def __len__(self):
        return len(self.chunks)

    # --- scoring ------------------------------------------------------------

    def _keyword_scores(self, query):
        """Rare query terms count for more than common ones.

        Not BM25 — no term frequency, no length normalisation — but preserved
        exactly as it was so the change to matrix maths cannot alter a ranking.

        This half is now the slow one: 2 ms against 0.1 ms for the cosine
        matmul, because substring matching means genuinely scanning every body
        for every term. Tokenising would make it a dictionary lookup, but
        "eco" would stop matching "ecology" and different chunks would come
        back. That is a retrieval change, so it waits for the eval set.
        """
        terms = query_terms(query)
        if not terms:
            return np.zeros(len(self.chunks))

        present = {t: np.fromiter((t in text for text in self._lower),
                                  dtype=bool, count=len(self._lower))
                   for t in terms}
        frequency = {t: int(mask.sum()) for t, mask in present.items()}
        highest = max(frequency.values()) or 1
        weights = {t: 1 - (f / highest) + 0.1 for t, f in frequency.items()}

        total = sum(weights.values()) or 1
        matched = np.zeros(len(self.chunks))
        for t, mask in present.items():
            matched += mask * weights[t]
        return matched / total

    def score(self, query, q_vec, fusion="weighted"):
        """Blended score per chunk, plus the two components."""
        q = np.asarray(q_vec, dtype=float)
        norm = np.linalg.norm(q) or 1.0
        cosine = self.vectors @ (q / norm)
        keyword = self._keyword_scores(query)

        if fusion == "weighted":
            blended = DENSE_WEIGHT * cosine + (1 - DENSE_WEIGHT) * keyword
        elif fusion == "rrf":
            blended = _rrf(cosine, keyword)
        else:
            raise ValueError(f"unknown fusion {fusion!r}")
        return blended, cosine, keyword

    def search(self, query, q_vec, top_k=10, fusion="weighted"):
        """Top chunks, at most one per document section."""
        if not self.chunks:
            return []
        blended, cosine, keyword = self.score(query, q_vec, fusion)

        best_by_section = {}
        for i in np.argsort(blended)[::-1][:CANDIDATES]:
            section = self.chunks[i].section
            if section not in best_by_section:
                best_by_section[section] = Hit(
                    chunk=self.chunks[i], score=float(blended[i]),
                    cosine=float(cosine[i]), keyword=float(keyword[i]), index=int(i))

        ranked = sorted(best_by_section.values(), key=lambda h: h.score, reverse=True)
        return ranked[:top_k]


def _rrf(*score_arrays, k=RRF_K):
    """Fuse on rank rather than score.

    The two signals are on different scales — cosine sits in a narrow band near
    the top, the keyword score spans zero to one — so adding them weighted is
    comparing units that do not match. Rank fusion sidesteps that entirely.
    """
    fused = np.zeros(len(score_arrays[0]))
    for scores in score_arrays:
        order = np.argsort(scores)[::-1]
        ranks = np.empty(len(scores), dtype=int)
        ranks[order] = np.arange(len(scores))
        fused += 1.0 / (k + ranks + 1)
    return fused


class Matcher:
    """Threshold match against a small set of records: publications, D2.2 sections.

    Same shape as the index but without the keyword half or the section
    deduplication, because these are matched purely on meaning.
    """

    def __init__(self, records, vectors, threshold):
        self.records = records
        self.vectors = normalise(np.asarray(vectors, dtype=float))
        self.threshold = threshold

    @classmethod
    def build(cls, records, texts, embed_fn, threshold):
        if not records:
            return cls([], np.zeros((0, 0)), threshold)
        return cls(records, embed_all(embed_fn, texts), threshold)

    def match(self, q_vec, limit=2):
        """Records above the threshold, best first, each with its score."""
        if not self.records or q_vec is None:
            return []
        q = np.asarray(q_vec, dtype=float)
        norm = np.linalg.norm(q) or 1.0
        scores = self.vectors @ (q / norm)
        order = [i for i in np.argsort(scores)[::-1] if scores[i] >= self.threshold]
        return [(self.records[i], float(scores[i])) for i in order[:limit]]


def references_for(hits, doc_references):
    """Reference lists belonging to the documents these hits came from."""
    sources = {h.chunk.source for h in hits}
    return {s: doc_references[s] for s in sources if doc_references.get(s)}
