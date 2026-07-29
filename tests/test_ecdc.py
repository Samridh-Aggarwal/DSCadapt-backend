"""Surveillance relevance, proved against the original implementation.

disease_relevant() from app.py is reproduced verbatim below. The new code must
agree with it on every disease and country the Atlas covers. It must disagree
in exactly one place — the three countries the Atlas does not cover — and that
disagreement is the bug being fixed.
"""

import json
import time
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
import ecdc  # noqa: E402
import ontology  # noqa: E402


# --- the 32 countries the interface offers ----------------------------------

UI_COUNTRIES = [
    "Austria", "Belgium", "Bulgaria", "Croatia", "Cyprus", "Czechia", "Denmark",
    "Estonia", "Finland", "France", "Germany", "Greece", "Hungary", "Iceland",
    "Ireland", "Italy", "Latvia", "Liechtenstein", "Lithuania", "Luxembourg",
    "Malta", "Netherlands", "Norway", "Poland", "Portugal", "Romania", "Slovakia",
    "Slovenia", "Spain", "Sweden", "Switzerland", "United Kingdom",
]

# the extract covers 29 of them
NOT_IN_EXTRACT = ["United Kingdom", "Switzerland", "Liechtenstein"]
EXTRACT_COUNTRIES = [c for c in UI_COUNTRIES if c not in NOT_IN_EXTRACT]

# realistic gaps so both branches of the case test get exercised
NO_CASES = {
    "CCHF": [c for c in EXTRACT_COUNTRIES if c not in ("Bulgaria", "Greece", "Spain")],
    "Hantavirus": ["Malta", "Cyprus"],
}


@pytest.fixture(scope="module")
def surv():
    rows = []
    for name, d in ecdc.DISEASES.items():
        if not d["ecdc"]:
            continue
        for country in EXTRACT_COUNTRIES:
            for year in range(2018, 2025):
                cases = 0 if country in NO_CASES.get(d["ecdc"], []) else 40 + year % 17
                rows.append({"disease": d["ecdc"], "country": country,
                             "country_code": country[:2].upper(), "year": year,
                             "cases": cases, "deaths": 2 if d["group"] == "1" else None,
                             "group": d["group"], "dark_figure": "High",
                             "eu_mandatory": "Yes", "coverage": "High"})
    return ecdc.Surveillance(pd.DataFrame(rows), loaded_at=time.time())


# --- verbatim from app.py ----------------------------------------------------

APP_AUTO_GATED = {"Chikungunya", "Dengue fever", "Malaria", "Zika fever", "West Nile fever"}

APP_AUTOCHTHONOUS = {
    "Dengue fever": {"France", "Italy", "Spain", "Croatia", "Portugal"},
    "Chikungunya": {"France", "Italy", "Spain"},
    "Zika fever": {"France"},
    "Malaria": {"Greece"},
    "West Nile fever": {"Italy", "Greece", "Romania", "Hungary", "Spain", "France",
                        "Germany", "Croatia", "Austria", "Bulgaria", "Cyprus",
                        "Slovenia", "Czechia", "Netherlands", "Slovakia"},
}

APP_ENDEMIC_2C = {
    "Leishmaniasis": {"Spain", "Portugal", "France", "Italy", "Greece", "Malta", "Cyprus", "Croatia", "Slovenia", "Bulgaria"},
    "Usutu virus": {"Italy", "France", "Germany", "Austria", "Czechia", "Hungary", "Croatia", "Spain", "Belgium", "Netherlands"},
    "Vibriosis": {"Germany", "Denmark", "Sweden", "Finland", "Poland", "Estonia", "Latvia", "Lithuania", "Netherlands", "Spain", "Italy", "Greece"},
}

APP_DISEASE_INFO = {n: {"ecdc": d["ecdc"], "group": d["group"]} for n, d in ecdc.DISEASES.items()}


def app_disease_relevant(disease, country, df):
    """The original, unchanged."""
    info = APP_DISEASE_INFO.get(disease)
    if info is None:
        return True
    if disease == "Rift Valley fever (RVF)":
        return False
    if not country or country == "All Europe":
        return True
    group = info["group"]
    if group in ("1", "2b"):
        return True
    if group == "2c":
        return country in APP_ENDEMIC_2C.get(disease, set())
    if group == "2a":
        if disease in APP_AUTO_GATED:
            return country in APP_AUTOCHTHONOUS.get(disease, set())
        if df is None:
            return True
        rows = df[(df["disease"] == info["ecdc"])
                  & (df["country"].str.lower() == country.lower())]
        return bool(rows["cases"].fillna(0).sum() > 0)
    return True


# --- the comparison ----------------------------------------------------------

def test_agrees_with_the_original_everywhere_the_atlas_has_data(surv):
    for disease in ecdc.DISEASES:
        for country in EXTRACT_COUNTRIES:
            assert surv.relevant(disease, country) == \
                   app_disease_relevant(disease, country, surv.df), \
                   f"{disease} / {country}"


def test_agrees_on_all_europe_and_on_the_reference_data(surv):
    for disease in ecdc.DISEASES:
        assert surv.relevant(disease, "All Europe") == \
               app_disease_relevant(disease, "All Europe", surv.df)
        info = ecdc.DISEASES[disease]
        assert info["group"] in ("1", "2a", "2b", "2c")
        assert info["gating"] in ("none", "autochthonous", "endemic", "never")
        if info["gating"] in ("autochthonous", "endemic"):
            assert info.get("countries"), f"{disease} is gated but lists no countries"


def test_the_only_disagreement_is_the_three_missing_countries(surv):
    differ = [(d, c) for d in ecdc.DISEASES for c in UI_COUNTRIES
              if surv.relevant(d, c) != app_disease_relevant(d, c, surv.df)]
    assert {c for _, c in differ} == set(NOT_IN_EXTRACT)
    assert {d for d, _ in differ} == {
        "Crimean-Congo haemorraghic fever (CCHF)", "Hantavirus infection",
        "Legionnaires' disease", "Leptospirosis"}, \
        "only group 2a diseases judged on case counts should be affected"
    assert len(differ) == 12, f"expected 3 countries x 4 diseases, got {len(differ)}"
    # every disagreement is the new code being permissive where the old one
    # inferred absence from missing data
    assert all(surv.relevant(d, c) for d, c in differ)


def test_uk_regains_the_diseases_it_was_silently_losing(surv):
    old = sum(app_disease_relevant(d, "United Kingdom", surv.df) for d in ecdc.DISEASES)
    new = len(surv.relevant_diseases("United Kingdom"))
    germany = len(surv.relevant_diseases("Germany"))
    assert old == 11, f"the original allowed {old} diseases for the UK"
    assert new == 15, f"the fix allows {new}"
    assert new > old
    for d in ["Legionnaires' disease", "Leptospirosis", "Hantavirus infection"]:
        assert not app_disease_relevant(d, "United Kingdom", surv.df)
        assert surv.relevant(d, "United Kingdom")
    assert germany >= new


def test_missing_country_is_not_the_same_as_zero_cases(surv):
    assert surv.has_country("Germany")
    assert not surv.has_country("United Kingdom")
    # Malta genuinely reports zero Hantavirus and must still be filtered out
    assert surv.total_cases("Hantavirus infection", "Malta") == 0
    assert not surv.relevant("Hantavirus infection", "Malta")


def test_a_failed_refresh_keeps_the_data_already_loaded(surv):
    """One bad monthly reload must not turn a working Space into one with no
    surveillance data until it restarts."""
    before = len(surv.df)
    surv.source = ("nonexistent/repo", "nope.csv")
    surv.load()                      # will fail: no such dataset
    assert surv.available and len(surv.df) == before
    assert surv.load_error


def test_a_first_load_that_fails_leaves_it_unavailable():
    down = ecdc.Surveillance(None, source=("nonexistent/repo", "nope.csv"))
    down.load()
    assert not down.available and down.load_error


def test_refresh_does_nothing_without_a_source():
    ecdc.Surveillance(None).refresh_if_stale()   # must not raise


def test_relevance_is_permissive_when_the_data_is_unavailable():
    down = ecdc.Surveillance(None)
    assert not down.available
    assert down.relevant("Legionnaires' disease", "Germany")
    assert not down.relevant("Rift Valley fever (RVF)", "Germany")
    assert not down.relevant("Dengue fever", "Finland")   # gating needs no data


# --- the context block -------------------------------------------------------

def test_block_is_empty_for_a_country_with_no_surveillance(surv):
    assert surv.block(["Legionnaires' disease"], "United Kingdom") == ""
    assert surv.block(["Legionnaires' disease"], "All Europe") == ""
    assert surv.block([], "Germany") == ""


def test_block_year_range_comes_from_the_data_not_a_literal(surv):
    text = surv.block(["Legionnaires' disease"], "Germany")
    assert "(2018-2024)" in text, "year range should follow the dataframe"
    assert "2018-2023" not in text


def test_block_renders_cases_deaths_and_the_group_note(surv):
    text = surv.block(["Salmonellosis"], "Germany")
    assert "Salmonellosis in Germany" in text
    assert "Reported cases:" in text and "2024:" in text
    assert "Deaths:" in text
    assert "Group 1:" in text
    assert "Under-reporting: High" in text


# --- the disease scan --------------------------------------------------------

def test_scan_reads_query_chunks_and_web_together():
    found = ecdc.scan_text("what about Italy",
                           "the retrieved passage mentions Legionella",
                           "recent web reports describe West Nile virus")
    assert found == ["Legionnaires' disease", "West Nile fever"]


def test_scan_of_web_text_alone_still_finds_diseases():
    """The original only ever scanned the query and the chunks, so a web-routed
    answer produced no surveillance data at all."""
    assert ecdc.scan_text("", "", "dengue cases reported in Croatia") == ["Dengue fever"]


def test_scan_respects_word_boundaries():
    assert ecdc.scan_text("the footbed was damp") == []
    assert ecdc.scan_text("TBE is expanding northwards") == ["Tick-borne encephalitis (TBE)"]
    assert ecdc.scan_text("salmonellosis outbreak") == ["Salmonellosis"]


def test_scan_ignores_diseases_absent_from_the_atlas():
    """Norovirus, Leishmaniasis, Usutu and Vibriosis carry no aliases on purpose,
    so no data block is ever built for them."""
    for name in ["Norovirus infection", "Leishmaniasis", "Usutu virus", "Vibriosis"]:
        assert ecdc.DISEASES[name]["aliases"] == []
    assert ecdc.scan_text("norovirus and leishmaniasis and usutu and vibriosis") == []


# --- reference data integrity ------------------------------------------------

def test_disease_names_match_the_ontology_exactly():
    assert set(ecdc.DISEASES) == set(ontology.DISEASES), (
        f"only in diseases.json: {sorted(set(ecdc.DISEASES) - set(ontology.DISEASES))}\n"
        f"only in the ontology:  {sorted(set(ontology.DISEASES) - set(ecdc.DISEASES))}")


def test_no_alias_is_claimed_by_two_diseases():
    seen = {}
    for name, d in ecdc.DISEASES.items():
        for alias in d["aliases"]:
            assert alias not in seen, f"{alias!r} claimed by {seen[alias]} and {name}"
            seen[alias] = name


def test_gated_countries_are_all_real():
    for name, d in ecdc.DISEASES.items():
        for c in d.get("countries", []):
            assert c in UI_COUNTRIES, f"{name} gates on {c!r}, which the interface never offers"
