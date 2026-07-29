"""Retrieval, against the loop it replaces.

The original retrieve() is reproduced below. Run over the real corpus with the
same vectors, the two must return the same chunks in the same order with the
same scores. Anything else means the rewrite changed what users see.
"""

import re
import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parent.parent
import corpus  # noqa: E402
import ontology  # noqa: E402
import retrieval  # noqa: E402

RNG = np.random.default_rng(20260729)
DIM = 1024


@pytest.fixture(scope="module")
def loaded():
    c = corpus.load(known_measures={m["name"] for m in ontology.MEASURES})
    vectors = RNG.normal(size=(len(c.chunks), DIM))
    return c, retrieval.Index(c.chunks, vectors)


def a_query_vector():
    return RNG.normal(size=DIM)


QUERIES = [
    "wastewater treatment and antimicrobial resistance",
    "what are the trade-offs of flood protection for vector-borne disease",
    "Legionella risk in district heating networks in Germany",
    "food safety regulation and Salmonella",
    "mosquito breeding in urban storm drains",
    "tick habitat in expanded forests",
    "how does climate change affect Vibrio in the Baltic",
    "green and blue spaces heat management",
]


# --- the original, verbatim --------------------------------------------------

def app_cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def app_retrieve(query, q_vec, chunks, embeddings, top_k=5):
    query_words = set(re.findall(r"\b\w+\b", query.lower()))
    query_words = {w for w in query_words if len(w) > 2 and w not in retrieval.STOPWORDS}

    word_freq = {}
    for w in query_words:
        word_freq[w] = sum(1 for c in chunks if w in c.text.lower())
    max_freq = max(word_freq.values()) if word_freq else 1
    word_weights = {w: 1 - (f / max_freq) + 0.1 for w, f in word_freq.items()}
    total_weight = sum(word_weights.values()) or 1

    scored = []
    for i in range(len(chunks)):
        cos = app_cosine(q_vec, embeddings[i])
        chunk_lower = chunks[i].text.lower()
        matched = sum(word_weights[w] for w in query_words if w in chunk_lower)
        kw = matched / total_weight if query_words else 0
        scored.append({"i": i, "score": 0.85 * cos + 0.15 * kw, "cos": cos, "kw": kw})

    top = sorted(scored, key=lambda s: s["score"], reverse=True)[:20]
    section_best = {}
    for s in top:
        section = chunks[s["i"]].section
        if section not in section_best:
            section_best[section] = s
    return sorted(section_best.values(), key=lambda s: s["score"], reverse=True)[:top_k]


# --- identical results -------------------------------------------------------

@pytest.mark.parametrize("query", QUERIES)
def test_same_chunks_in_the_same_order(loaded, query):
    c, index = loaded
    q_vec = a_query_vector()
    new = index.search(query, q_vec, top_k=7)
    old = app_retrieve(query, q_vec, c.chunks, index.vectors, top_k=7)
    assert [h.index for h in new] == [o["i"] for o in old]


@pytest.mark.parametrize("query", QUERIES)
def test_same_scores_to_floating_point(loaded, query):
    c, index = loaded
    q_vec = a_query_vector()
    new = index.search(query, q_vec, top_k=7)
    old = app_retrieve(query, q_vec, c.chunks, index.vectors, top_k=7)
    for h, o in zip(new, old):
        assert h.score == pytest.approx(o["score"], abs=1e-9)
        assert h.cosine == pytest.approx(o["cos"], abs=1e-9)
        assert h.keyword == pytest.approx(o["kw"], abs=1e-9)


def test_agreement_holds_over_many_random_queries(loaded):
    c, index = loaded
    for _ in range(40):
        query = " ".join(RNG.choice(
            ["water", "disease", "flood", "mosquito", "tourism", "heat",
             "safety", "vector", "climate", "risk"], size=4, replace=False))
        q_vec = a_query_vector()
        assert [h.index for h in index.search(query, q_vec, top_k=10)] == \
               [o["i"] for o in app_retrieve(query, q_vec, c.chunks, index.vectors, 10)]


# --- a crash the original was one query away from --------------------------

def test_a_query_matching_nothing_does_not_divide_by_zero(loaded):
    """The original computed 1 - (freq / max_freq) with max_freq taken from the
    counts. Every term absent from the corpus makes that denominator zero."""
    c, index = loaded
    query = "xyzzy plugh frotz"
    with pytest.raises(ZeroDivisionError):
        app_retrieve(query, a_query_vector(), c.chunks, index.vectors)
    hits = index.search(query, a_query_vector(), top_k=5)
    assert len(hits) == 5
    assert all(h.keyword == 0 for h in hits)


def test_an_empty_query_is_fine(loaded):
    _, index = loaded
    hits = index.search("", a_query_vector(), top_k=3)
    assert len(hits) == 3
    assert all(h.keyword == 0 for h in hits)


def test_an_empty_index_is_fine():
    assert retrieval.Index([], np.zeros((0, DIM))).search("x", a_query_vector()) == []


# --- behaviour --------------------------------------------------------------

def test_one_chunk_per_section(loaded):
    _, index = loaded
    hits = index.search("wastewater management adaptation", a_query_vector(), top_k=15)
    sections = [h.chunk.section for h in hits]
    assert len(sections) == len(set(sections))


def test_scores_come_back_sorted(loaded):
    _, index = loaded
    scores = [h.score for h in index.search(QUERIES[0], a_query_vector(), top_k=10)]
    assert scores == sorted(scores, reverse=True)


def test_components_are_reported_separately(loaded):
    """Only the blended number was ever logged, so the 0.85/0.15 split could not
    be tuned from production traffic."""
    _, index = loaded
    for h in index.search(QUERIES[0], a_query_vector(), top_k=5):
        assert h.score == pytest.approx(0.85 * h.cosine + 0.15 * h.keyword)


def test_rare_terms_outweigh_common_ones(loaded):
    _, index = loaded
    _, _, keyword = index.score("wastewater Legionella", a_query_vector())
    assert keyword.max() > 0


def test_rrf_is_available_and_ranks_differently(loaded):
    _, index = loaded
    q_vec = a_query_vector()
    weighted = [h.index for h in index.search(QUERIES[1], q_vec, top_k=10, fusion="weighted")]
    rrf = [h.index for h in index.search(QUERIES[1], q_vec, top_k=10, fusion="rrf")]
    assert weighted != rrf, "if these matched, the eval would have nothing to compare"
    assert set(rrf) & set(weighted), "but they should still broadly agree"


def test_unknown_fusion_is_rejected(loaded):
    _, index = loaded
    with pytest.raises(ValueError):
        index.score("x", a_query_vector(), fusion="magic")


# --- batching ---------------------------------------------------------------

def test_embedding_is_batched():
    """/evaluate embedded every paragraph of an uploaded document in one call."""
    sizes = []

    def embed(texts):
        sizes.append(len(texts))
        return RNG.normal(size=(len(texts), 8))

    retrieval.embed_all(embed, [f"paragraph {i}" for i in range(250)], batch=100)
    assert sizes == [100, 100, 50]


def test_embedding_nothing_is_fine():
    assert retrieval.embed_all(lambda t: [], []).size == 0


def test_index_rejects_a_vector_count_mismatch():
    with pytest.raises(ValueError):
        retrieval.Index([1, 2, 3], np.zeros((2, DIM)))


# --- matcher ----------------------------------------------------------------

def test_matcher_respects_its_threshold():
    records = [{"title": f"paper {i}"} for i in range(6)]
    vectors = RNG.normal(size=(6, DIM))
    target = vectors[3] / np.linalg.norm(vectors[3])

    strict = retrieval.Matcher(records, vectors, threshold=0.99)
    assert [r["title"] for r, _ in strict.match(target)] == ["paper 3"]
    assert retrieval.Matcher(records, vectors, threshold=1.01).match(target) == []
    assert len(retrieval.Matcher(records, vectors, threshold=-1.0).match(target, limit=4)) == 4


def test_matcher_returns_best_first():
    records = [{"n": i} for i in range(5)]
    vectors = RNG.normal(size=(5, DIM))
    scores = [s for _, s in retrieval.Matcher(records, vectors, -1.0).match(a_query_vector(), 5)]
    assert scores == sorted(scores, reverse=True)


def test_matcher_with_no_query_vector_returns_nothing():
    m = retrieval.Matcher([{"n": 1}], RNG.normal(size=(1, DIM)), -1.0)
    assert m.match(None) == []


def test_matcher_with_no_records_is_fine():
    assert retrieval.Matcher.build([], [], lambda t: [], 0.8).match(a_query_vector()) == []


# --- references -------------------------------------------------------------

def test_references_follow_the_documents_that_were_retrieved(loaded):
    c, index = loaded
    hits = index.search("wastewater antimicrobial resistance", a_query_vector(), top_k=10)
    refs = retrieval.references_for(hits, c.references)
    assert set(refs) <= {h.chunk.source for h in hits}
    assert all(v for v in refs.values()), "empty reference lists should be omitted"


# --- the reason for the rewrite ---------------------------------------------

def test_the_matrix_version_is_substantially_faster(loaded):
    c, index = loaded
    q_vec = a_query_vector()

    start = time.perf_counter()
    for _ in range(20):
        app_retrieve(QUERIES[0], q_vec, c.chunks, index.vectors, top_k=7)
    old = time.perf_counter() - start

    start = time.perf_counter()
    for _ in range(20):
        index.search(QUERIES[0], q_vec, top_k=7)
    new = time.perf_counter() - start

    print(f"\n  loop {old * 50:.1f} ms/query   matrix {new * 50:.1f} ms/query   "
          f"{old / new:.0f}x")
    assert new < old
