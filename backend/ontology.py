"""The measure -> mechanism -> category -> disease graph.

Loaded from data/ontology.json, which ontology/build.py generates from the
authored diagram. Nothing here is hand-written domain knowledge; this module
only reads the graph and derives the two views the rest of the backend wants:

    MEASURE_RISKS      measure -> [(mechanism, category)]  trade-off edges only
    CATEGORY_DISEASES  category -> [disease]               parent categories resolved

Both used to be hardcoded dicts in app.py. They were faithful to the diagram,
but they were a third hand-maintained copy of it alongside the CSV and the
frontend, and the three had already drifted on mechanism wording.
"""

import json
from pathlib import Path

DATA = Path(__file__).parent / "data" / "ontology.json"

_g = json.loads(DATA.read_text(encoding="utf-8"))

SECTORS = _g["sectors"]
MEASURES = _g["measures"]
MECHANISM_TYPE = _g["mechanisms"]
CATEGORIES = _g["categories"]

_EDGES = _g["edges"]

# measure id <-> canonical name
ID_BY_MEASURE = {m["name"]: m["id"] for m in MEASURES}
MEASURE_BY_ID = {m["id"]: m["name"] for m in MEASURES}


def _direct_leaves():
    out = {}
    for cat, disease in _EDGES["category_disease"]:
        out.setdefault(cat, []).append(disease)
    return out


def _sub_categories():
    out = {}
    for parent, child in _EDGES["category_category"]:
        out.setdefault(parent, []).append(child)
    return out


def _resolve(category, leaves, subs, seen=None):
    """Diseases under a category, following sub-categories.

    Vector-borne diseases has no diseases of its own — it expands into
    mosquito-, tick- and other vector-borne, which do.
    """
    seen = seen or set()
    if category in seen:
        raise ValueError(f"cycle in the category graph at {category!r}")
    seen = seen | {category}
    found = set(leaves.get(category, []))
    for child in subs.get(category, []):
        found |= _resolve(child, leaves, subs, seen)
    return found


_leaves, _subs = _direct_leaves(), _sub_categories()

CATEGORY_DISEASES = {
    c: sorted(_resolve(c, _leaves, _subs)) for c in CATEGORIES
}

DISEASES = sorted({d for ds in CATEGORY_DISEASES.values() for d in ds})

# A category with no diseases is a terminal that deliberately names none.
# Human susceptibility is the one that matters: eight mechanisms route to it
# because they raise general vulnerability without selecting for a pathogen.
TERMINAL_CATEGORIES = sorted(c for c, ds in CATEGORY_DISEASES.items() if not ds)


def _measure_risks():
    """measure -> [(mechanism, category)] for trade-off edges only.

    Measures with no trade-off edge are absent, which is how the reviewer knows
    never to comment on them.
    """
    by_mechanism = {}
    for mech, cat in _EDGES["mechanism_category"]:
        by_mechanism.setdefault(mech, []).append(cat)

    out = {}
    for measure, mech, kind in _EDGES["measure_mechanism"]:
        if kind != "trade-off":
            continue
        for cat in by_mechanism.get(mech, []):
            out.setdefault(measure, []).append((mech, cat))
    return {m: sorted(set(pairs)) for m, pairs in out.items()}


MEASURE_RISKS = _measure_risks()


def mechanisms_for(measure, kind=None):
    return sorted({mech for m, mech, k in _EDGES["measure_mechanism"]
                   if m == measure and (kind is None or k == kind)})


def diseases_for(measure):
    """Every disease reachable from a measure's trade-off edges."""
    return sorted({d for _, cat in MEASURE_RISKS.get(measure, [])
                   for d in CATEGORY_DISEASES[cat]})


def validate():
    """Structural checks. Raises on anything that would fail silently at runtime."""
    problems = []

    ids = [m["id"] for m in MEASURES]
    if len(ids) != len(set(ids)):
        problems.append("duplicate measure ids")
    names = [m["name"] for m in MEASURES]
    if len(names) != len(set(names)):
        problems.append("duplicate measure names")

    sector_ids = {s["id"] for s in SECTORS}
    for m in MEASURES:
        if m["sector"] not in sector_ids:
            problems.append(f"{m['id']} points at unknown sector {m['sector']!r}")

    known = set(MECHANISM_TYPE)
    for _, mech, _ in _EDGES["measure_mechanism"]:
        if mech not in known:
            problems.append(f"untyped mechanism {mech!r}")
    for mech, cat in _EDGES["mechanism_category"]:
        if mech not in known:
            problems.append(f"category edge from unknown mechanism {mech!r}")
        if cat not in CATEGORIES:
            problems.append(f"unknown category {cat!r}")

    # A measure reaching no disease is only wrong if one of its categories was
    # supposed to hold some. Measures whose trade-offs all land on a terminal
    # (Marine spatial planning -> social unrest -> human susceptibility) are
    # correct and produce no comment by design.
    for measure, pairs in MEASURE_RISKS.items():
        cats = {c for _, c in pairs}
        if not diseases_for(measure) and not cats <= set(TERMINAL_CATEGORIES):
            live = sorted(cats - set(TERMINAL_CATEGORIES))
            problems.append(f"{measure!r} reaches no disease via non-terminal {live}")

    if problems:
        raise ValueError("ontology validation failed:\n  " + "\n  ".join(problems))
    return True
