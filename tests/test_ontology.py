"""The generated ontology must equal the CSV and equal what app.py has today.

The second half matters more than the first. MEASURE_RISKS and CATEGORY_DISEASES
below are copied verbatim out of the current app.py. If the generated graph
reproduces them exactly, moving the graph into a data file provably changed no
behaviour — which is the whole point of doing it before touching anything else.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
import ontology  # noqa: E402


# --- verbatim from app.py, do not tidy --------------------------------------

APP_MEASURE_RISKS = {
    "Water storage, distribution & reuse": [("Habitat for disease pathogens & vectors", "Vector-borne diseases"), ("Habitat for disease pathogens & vectors", "Animal contact (direct or indirect)")],
    "Water quality monitoring & improvement": [("Ecological disruption", "Vector-borne diseases"), ("Ecological disruption", "Animal contact (direct or indirect)")],
    "Flood protection": [("Ecological disruption", "Vector-borne diseases"), ("Ecological disruption", "Animal contact (direct or indirect)"), ("Habitat for disease pathogens & vectors", "Vector-borne diseases"), ("Habitat for disease pathogens & vectors", "Animal contact (direct or indirect)"), ("Social unrest, displacement & mental health risks", "Human susceptibility to infectious disease")],
    "Marine spatial planning": [("Social unrest, displacement & mental health risks", "Human susceptibility to infectious disease")],
    "Fishing rules & improve fish health": [("Ecological disruption", "Vector-borne diseases"), ("Ecological disruption", "Animal contact (direct or indirect)")],
    "Sustainable farming strategies": [("Habitat for disease pathogens & vectors", "Vector-borne diseases"), ("Habitat for disease pathogens & vectors", "Animal contact (direct or indirect)")],
    "Irrigation systems": [("Ecological disruption", "Vector-borne diseases"), ("Ecological disruption", "Animal contact (direct or indirect)")],
    "Protect and restore biodiversity, habitats & natural resources": [("Ecological disruption", "Vector-borne diseases"), ("Ecological disruption", "Animal contact (direct or indirect)"), ("Increased human presence in natural areas", "Vector-borne diseases"), ("Increased human presence in natural areas", "Animal contact (direct or indirect)"), ("Land use conflict", "Human susceptibility to infectious disease")],
    "Expand forested and natural areas": [("Ecological disruption", "Vector-borne diseases"), ("Ecological disruption", "Animal contact (direct or indirect)"), ("Increased human presence in natural areas", "Vector-borne diseases"), ("Increased human presence in natural areas", "Animal contact (direct or indirect)"), ("Land use conflict", "Human susceptibility to infectious disease")],
    "Increase green & blue spaces": [("Allergies & respiratory issues", "Air-borne diseases"), ("Allergies & respiratory issues", "Human susceptibility to infectious disease"), ("Habitat for disease pathogens & vectors", "Vector-borne diseases"), ("Habitat for disease pathogens & vectors", "Animal contact (direct or indirect)")],
    "Retrofit and insulate buildings": [("Allergies & respiratory issues", "Air-borne diseases"), ("Allergies & respiratory issues", "Human susceptibility to infectious disease"), ("Overheating", "Human susceptibility to infectious disease")],
    "Expand hydropower": [("Ecological disruption", "Vector-borne diseases"), ("Ecological disruption", "Animal contact (direct or indirect)"), ("Habitat for disease pathogens & vectors", "Vector-borne diseases"), ("Habitat for disease pathogens & vectors", "Animal contact (direct or indirect)")],
    "Expand public transport": [("Increased contact rates & overcrowding", "Air-borne diseases")],
    "Expand tourism seasons": [("Difficulty in detecting diseases in returning travelers", "Vector-borne diseases"), ("Difficulty in detecting diseases in returning travelers", "Water-borne diseases"), ("Difficulty in detecting diseases in returning travelers", "Food-borne diseases"), ("Difficulty in detecting diseases in returning travelers", "Animal contact (direct or indirect)"), ("Difficulty in detecting diseases in returning travelers", "Air-borne diseases"), ("Shift in disease patterns", "Vector-borne diseases"), ("Shift in disease patterns", "Water-borne diseases"), ("Shift in disease patterns", "Food-borne diseases"), ("Shift in disease patterns", "Animal contact (direct or indirect)"), ("Shift in disease patterns", "Air-borne diseases"), ("Social unrest, displacement & mental health risks", "Human susceptibility to infectious disease")],
    "Expand tourism in new locations": [("Difficulty in detecting diseases in returning travelers", "Vector-borne diseases"), ("Difficulty in detecting diseases in returning travelers", "Water-borne diseases"), ("Difficulty in detecting diseases in returning travelers", "Food-borne diseases"), ("Difficulty in detecting diseases in returning travelers", "Animal contact (direct or indirect)"), ("Difficulty in detecting diseases in returning travelers", "Air-borne diseases"), ("Ecological disruption", "Vector-borne diseases"), ("Ecological disruption", "Animal contact (direct or indirect)"), ("Limited capacity of healthcare system", "Human susceptibility to infectious disease"), ("Shift in disease patterns", "Vector-borne diseases"), ("Shift in disease patterns", "Water-borne diseases"), ("Shift in disease patterns", "Food-borne diseases"), ("Shift in disease patterns", "Animal contact (direct or indirect)"), ("Shift in disease patterns", "Air-borne diseases"), ("Social unrest, displacement & mental health risks", "Human susceptibility to infectious disease")],
    "Improve infrastructures & resources for tourists": [("Ecological disruption", "Vector-borne diseases"), ("Ecological disruption", "Animal contact (direct or indirect)"), ("Social unrest, displacement & mental health risks", "Human susceptibility to infectious disease")],
}

APP_CATEGORY_DISEASES = {
    "Air-borne diseases": ['Hantavirus infection'],
    "Animal contact (direct or indirect)": ['Giardiasis', 'Hantavirus infection', 'Leptospirosis', 'Rift Valley fever (RVF)'],
    "Food-borne diseases": ['Campylobacteriosis', 'Crimean-Congo haemorraghic fever (CCHF)', 'Cryptosporidiosis', 'E. coli infection', 'Giardiasis', 'Hepatitis A', 'Listeriosis', 'Norovirus infection', 'Rift Valley fever (RVF)', 'Salmonellosis', 'Shigellosis', 'Tick-borne encephalitis (TBE)', 'Vibriosis'],
    "Human susceptibility to infectious disease": [],
    "Vector-borne diseases": ['Chikungunya', 'Crimean-Congo haemorraghic fever (CCHF)', 'Dengue fever', 'Leishmaniasis', 'Lyme disease', 'Malaria', 'Rift Valley fever (RVF)', 'Tick-borne encephalitis (TBE)', 'Usutu virus', 'West Nile fever', 'Zika fever'],
    "Water-borne diseases": ['Cryptosporidiosis', 'E. coli infection', 'Giardiasis', 'Hepatitis A', "Legionnaires' disease", 'Norovirus infection', 'Shigellosis', 'Vibriosis'],
}


# --- the generated files must still match the CSV ---------------------------

def test_build_is_reproducible(tmp_path):
    """Re-running the build must not change either output."""
    before = {p: p.read_bytes() for p in
              [ROOT / "backend/data/ontology.json", ROOT / "data/pathways.json"]}
    subprocess.run([sys.executable, str(ROOT / "ontology/build.py")], check=True,
                   capture_output=True)
    for path, original in before.items():
        assert path.read_bytes() == original, f"{path.name} changed on rebuild"


def test_ontology_is_valid():
    assert ontology.validate()


# --- the derived views must equal what app.py has today ---------------------

def test_measure_risks_matches_app():
    generated = {m: sorted(pairs) for m, pairs in ontology.MEASURE_RISKS.items()}
    expected = {m: sorted(pairs) for m, pairs in APP_MEASURE_RISKS.items()}
    assert set(generated) == set(expected), (
        f"measures only in generated: {sorted(set(generated) - set(expected))}\n"
        f"measures only in app.py:    {sorted(set(expected) - set(generated))}")
    for measure in expected:
        assert generated[measure] == expected[measure], f"edges differ for {measure!r}"


def test_category_diseases_matches_app():
    for category, diseases in APP_CATEGORY_DISEASES.items():
        assert category in ontology.CATEGORY_DISEASES, f"{category!r} missing"
        assert ontology.CATEGORY_DISEASES[category] == sorted(diseases), \
            f"diseases differ for {category!r}"


def test_generated_graph_reaches_more_than_app_did():
    """The CSV has categories app.py never modelled. They should now be present."""
    extra = set(ontology.CATEGORY_DISEASES) - set(APP_CATEGORY_DISEASES)
    assert extra == {"Anti-microbial resistance", "Mosquito-borne diseases",
                     "Other vector-borne diseases", "Tick-borne diseases"}


# --- structural properties --------------------------------------------------

def test_vector_borne_resolves_through_sub_categories():
    """Vector-borne has no diseases of its own; it expands into three children."""
    leaves = [d for c, d in ontology._EDGES["category_disease"] if c == "Vector-borne diseases"]
    assert leaves == [], "Vector-borne diseases should have no direct leaves"
    children = ontology.CATEGORY_DISEASES["Mosquito-borne diseases"] \
             + ontology.CATEGORY_DISEASES["Tick-borne diseases"] \
             + ontology.CATEGORY_DISEASES["Other vector-borne diseases"]
    assert ontology.CATEGORY_DISEASES["Vector-borne diseases"] == sorted(set(children))


def test_human_susceptibility_is_a_deliberate_terminal():
    """Eight mechanisms route here. It names no disease, and that is correct —
    raising general vulnerability does not select for a pathogen."""
    assert ontology.CATEGORY_DISEASES["Human susceptibility to infectious disease"] == []
    assert "Human susceptibility to infectious disease" in ontology.TERMINAL_CATEGORIES
    routed = {m for m, c in ontology._EDGES["mechanism_category"]
              if c == "Human susceptibility to infectious disease"}
    assert len(routed) >= 8


def test_measures_reach_a_disease_unless_every_category_is_terminal():
    """Reaching nothing is fine only when all the categories are terminals."""
    terminals = set(ontology.TERMINAL_CATEGORIES)
    for measure, pairs in ontology.MEASURE_RISKS.items():
        if ontology.diseases_for(measure):
            continue
        cats = {c for _, c in pairs}
        assert cats <= terminals, (
            f"{measure!r} reaches no disease via non-terminal {sorted(cats - terminals)}")


def test_measure_ids_are_unique_and_sorted():
    ids = [m["id"] for m in ontology.MEASURES]
    assert len(ids) == len(set(ids))
    assert ids == sorted(ids, key=lambda i: [int(p) for p in i.split(".")])


def test_tourism_is_sector_five_and_energy_is_seven():
    by_id = {m["id"]: m for m in ontology.MEASURES}
    assert by_id["5.1"]["name"] == "Expand tourism seasons"
    assert by_id["5.1"]["sector"] == "tourism"
    assert by_id["7.1"]["name"] == "Expand hydropower"
    assert by_id["7.1"]["sector"] == "energy"


def test_frontend_and_backend_describe_the_same_graph():
    """Same 92 edges, expressed with different labels."""
    pathways = json.loads((ROOT / "data/pathways.json").read_text(encoding="utf-8"))
    ui_edges = sum(len(v) for v in pathways["measureMechanisms"].values())
    assert ui_edges == len(ontology._EDGES["measure_mechanism"]) == 92
    assert len(pathways["measures"]) == len(ontology.MEASURES) == 26
    assert {s["id"] for s in pathways["sectors"]} == {s["id"] for s in ontology.SECTORS}


@pytest.mark.parametrize("measure,expected", [
    ("Expand public transport", ["Hantavirus infection"]),
    ("Marine spatial planning", []),
])
def test_known_walks(measure, expected):
    """Two the reviewer gets wrong in ways worth pinning down.

    Public transport crowding resolves to a rodent-aerosol disease because
    Air-borne holds only Hantavirus. Marine spatial planning reaches nothing
    because its one mechanism is a susceptibility terminal. Both are faithful
    to the diagram; both are worth a domain decision later.
    """
    assert ontology.diseases_for(measure) == expected
