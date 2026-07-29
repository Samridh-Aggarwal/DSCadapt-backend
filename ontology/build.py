"""Turn the authored pathway diagram into the two files the code reads.

pathway_diagram.csv is exported from the Miro diagram and is the single source
of truth for the whole measure -> mechanism -> category -> disease graph. This
script produces:

    backend/data/ontology.json   full graph, five levels, canonical names
    data/pathways.json           same graph with the short labels the explorer uses

Neither output should ever be hand-edited. Re-run this after the diagram
changes; tests/test_ontology.py fails if either file drifts from the CSV.

The only thing this file adds that is not in the CSV is presentation: measure
numbering and the shortened names the UI displays. Those are decisions about
the interface, not about the domain, which is why they live here rather than
in the diagram.
"""

import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent
CSV_PATH = HERE / "pathway_diagram.csv"

# --- presentation only -------------------------------------------------------

# Measure numbering follows the source document filenames, not CSV order.
# Tourism is sector 5 and energy sector 7, so hydropower is 7.1 even though its
# file is still named 5.1 — that file gets renumbered when the corpus is cleaned.
SECTORS = {
    "Water management":                            ("water",     "Water management"),
    "Coastal, marine & fisheries":                 ("coastal",   "Coastal & marine"),
    "Agriculture & food, biodiversity & forestry": ("food",      "Agriculture & food"),
    "Buildings & urban":                           ("buildings", "Buildings & urban"),
    "Tourism":                                     ("tourism",   "Tourism"),
    "Transport":                                   ("transport", "Transport"),
    "Energy":                                      ("energy",    "Energy"),
}

MEASURES = {
    "Water storage, distribution & reuse":                            ("1.1", "Water storage & reuse"),
    "Wastewater management":                                          ("1.2", "Wastewater management"),
    "Water quality monitoring & improvement":                         ("1.3", "Water quality monitoring"),
    "Awareness raising & information sharing":                        ("1.4", "Awareness & information"),
    "Early warning system for flood risk":                            ("1.5", "Early warning systems"),
    "Flood protection":                                               ("1.6", "Flood protection"),
    "Marine spatial planning":                                        ("2.1", "Marine spatial planning"),
    "Ocean acidification reduction":                                  ("2.2", "Reduce ocean acidification"),
    "Fishing rules & improve fish health":                            ("2.3", "Fishing rules & fish health"),
    "Sustainable farming strategies":                                 ("3.1", "Sustainable farming"),
    "Irrigation systems":                                             ("3.2", "Irrigation systems"),
    "Protect and restore biodiversity, habitats & natural resources": ("3.3", "Protect biodiversity"),
    "Expand forested and natural areas":                              ("3.4", "Expand forested areas"),
    "Promote plant-based diets":                                      ("3.5", "Plant-based diets"),
    "Improved food safety systems and regulations":                   ("3.6", "Food safety systems"),
    "Food safety education and outreach":                             ("3.7", "Food safety education"),
    "Increase green & blue spaces":                                   ("4.1", "Green & blue spaces"),
    "Building standards and regulations":                             ("4.2", "Building standards"),
    "Retrofit and insulate buildings":                                ("4.3", "Retrofitting & insulation"),
    "Expand tourism seasons":                                         ("5.1", "Expand tourism seasons"),
    "Expand tourism in new locations":                                ("5.2", "Tourism in new locations"),
    "Improve infrastructures & resources for tourists":               ("5.3", "Tourist infrastructure"),
    "Expand public transport":                                        ("6.1", "Public transport"),
    "Expand electrification":                                         ("6.2", "Electrification"),
    "Promote active mobility":                                        ("6.3", "Active mobility"),
    "Expand hydropower":                                              ("7.1", "Hydropower expansion"),
}

CATEGORIES = {
    "Air-borne diseases":                         "Air-borne",
    "Animal contact (direct or indirect)":        "Animal contact",
    "Anti-microbial resistance":                  "Anti-microbial resistance",
    "Food-borne diseases":                        "Food-borne",
    "Human susceptibility to infectious disease": "Human susceptibility",
    "Mosquito-borne diseases":                    "Mosquito-borne",
    "Other vector-borne diseases":                "Other vector-borne",
    "Tick-borne diseases":                        "Tick-borne",
    "Vector-borne diseases":                      "Vector-borne",
    "Water-borne diseases":                       "Water-borne",
}

# The explorer collapses this hop: it wires mechanisms straight to the three
# sub-categories rather than making the user click through the parent.
FLATTEN_IN_UI = {"Vector-borne diseases"}


def die(msg):
    sys.exit(f"build.py: {msg}")


def read_csv():
    """Split the CSV into its four labelled edge sections."""
    sections, current = {}, None
    for row in csv.reader(CSV_PATH.open(encoding="utf-8-sig")):
        row = (row + ["", "", ""])[:3]
        a, b, kind = (x.strip() for x in row)
        if a and not b and "-->" in a:
            current = a
            sections[current] = []
            continue
        if not a or not b or a in ("From", "Pathway diagram table"):
            continue
        if current is None:
            die(f"edge before any section header: {a} -> {b}")
        sections[current].append((a, b, kind))
    if len(sections) != 4:
        die(f"expected 4 sections in the CSV, found {len(sections)}: {list(sections)}")
    return list(sections.values())


def build():
    sector_measure, measure_mechanism, mech_to_cat, category_disease = read_csv()

    mechanism_type = {}
    for _, mech, kind in measure_mechanism:
        kind = {"Co-benefit": "co-benefit", "Trade-off": "trade-off"}.get(kind)
        if kind is None:
            die(f"mechanism {mech!r} has an unrecognised link type")
        mechanism_type.setdefault(mech, kind)

    # The third section mixes two edge kinds: mechanism -> category, and
    # category -> sub-category. Tell them apart by whether the source is a
    # mechanism.
    mechanism_category, category_category = [], []
    for src, dst, _ in mech_to_cat:
        (mechanism_category if src in mechanism_type else category_category).append([src, dst])

    # fail loudly rather than silently emitting a broken UI
    for name in {a for a, _, _ in sector_measure}:
        if name not in SECTORS:
            die(f"sector {name!r} is in the CSV but has no entry in SECTORS")
    for name in {b for _, b, _ in sector_measure}:
        if name not in MEASURES:
            die(f"measure {name!r} is in the CSV but has no entry in MEASURES")
    for name in {b for _, b in mechanism_category} | {a for a, _ in category_category} \
              | {b for a, b in category_category} | {a for a, _, _ in category_disease}:
        if name not in CATEGORIES:
            die(f"category {name!r} is in the CSV but has no entry in CATEGORIES")
    for src, _ in category_category:
        if src not in {b for _, b in mechanism_category}:
            die(f"category {src!r} has sub-categories but nothing points at it")

    sectors = [{"id": SECTORS[n][0], "name": SECTORS[n][1]}
               for n in dict.fromkeys(a for a, _, _ in sector_measure)]
    measures = [{"id": MEASURES[m][0], "name": m, "sector": SECTORS[s][0]}
                for s, m, _ in sector_measure]
    measures.sort(key=lambda x: [int(p) for p in x["id"].split(".")])

    ontology = {
        "source": "ontology/pathway_diagram.csv",
        "note": "Generated by ontology/build.py. Do not edit by hand.",
        "sectors": sectors,
        "measures": measures,
        "mechanisms": mechanism_type,
        "categories": sorted(CATEGORIES),
        "edges": {
            "measure_mechanism": [
                [m, mech, {"Co-benefit": "co-benefit", "Trade-off": "trade-off"}[k]]
                for m, mech, k in measure_mechanism],
            "mechanism_category": mechanism_category,
            "category_category": category_category,
            "category_disease": [[c, d] for c, d, _ in category_disease],
        },
    }

    # --- UI view: short labels, parent categories collapsed -------------------
    by_mech = {}
    for mech, cat in mechanism_category:
        by_mech.setdefault(mech, []).append(cat)
    subs = {}
    for parent, child in category_category:
        subs.setdefault(parent, []).append(child)

    def ui_categories(mech):
        out = []
        for cat in by_mech.get(mech, []):
            out.extend(subs[cat] if cat in FLATTEN_IN_UI else [cat])
        return sorted({CATEGORIES[c] for c in out})

    measure_mechanisms = {}
    for m, mech, kind in measure_mechanism:
        measure_mechanisms.setdefault(MEASURES[m][0], []).append(
            {"mechanism": mech, "type": "co" if kind == "Co-benefit" else "tr"})

    pathways = {
        "source": "ontology/pathway_diagram.csv",
        "note": "Generated by ontology/build.py. Do not edit by hand.",
        "sectors": sectors,
        "measures": [{"id": m["id"], "sector": m["sector"], "name": MEASURES[m["name"]][1]}
                     for m in measures],
        "mechanisms": {
            "coBenefit": sorted(m for m, t in mechanism_type.items() if t == "co-benefit"),
            "tradeOff":  sorted(m for m, t in mechanism_type.items() if t == "trade-off"),
        },
        "measureMechanisms": measure_mechanisms,
        "mechanismCategories": {m: ui_categories(m) for m in sorted(mechanism_type)},
        "categories": [CATEGORIES[c] for c in sorted(CATEGORIES)
                       if c not in FLATTEN_IN_UI and c != "Human susceptibility to infectious disease"],
    }

    return ontology, pathways


def main():
    ontology, pathways = build()
    for path, data in [(ROOT / "backend/data/ontology.json", ontology),
                       (ROOT / "data/pathways.json", pathways)]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote {path.relative_to(ROOT)}")

    e = ontology["edges"]
    print(f"  {len(ontology['sectors'])} sectors, {len(ontology['measures'])} measures, "
          f"{len(ontology['mechanisms'])} mechanisms, {len(ontology['categories'])} categories")
    print(f"  edges: {len(e['measure_mechanism'])} measure-mechanism, "
          f"{len(e['mechanism_category'])} mechanism-category, "
          f"{len(e['category_category'])} category-category, "
          f"{len(e['category_disease'])} category-disease")


if __name__ == "__main__":
    main()
