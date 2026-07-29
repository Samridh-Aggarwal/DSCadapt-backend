"""ECDC surveillance data, and whether a disease is worth naming for a country.

Three jobs:
  1. hold the published CSV and keep it fresh
  2. decide whether a disease is relevant to a given country
  3. render the context block the model reads

The disease reference — ECDC names, reliability groups, geographic gating,
text aliases — lives in data/diseases.json rather than in this file, so an
epidemiologist can correct it without touching Python.

The one behavioural change from the original app.py is in relevant(): a
country the Atlas does not cover is no longer read as a country with zero
cases. The UK, Switzerland and Liechtenstein are offered in the interface but
absent from the 29-country extract, so the old test silently declared
Legionnaires' disease and Leptospirosis "not established" there.
"""

import json
import re
import time
from pathlib import Path

import pandas as pd

DATA = Path(__file__).parent / "data" / "diseases.json"
REFRESH_AFTER = 30 * 24 * 3600
RETRY_AFTER = 300          # floor on retries after a failed load

_ref = json.loads(DATA.read_text(encoding="utf-8"))

GROUP_NOTES = _ref["groups"]
DISEASES = {d["name"]: d for d in _ref["diseases"]}
ECDC_NAME = {n: d["ecdc"] for n, d in DISEASES.items()}


def _alias_pattern(alias):
    """Word-bounded so 'tbe' does not match inside 'footbed'."""
    body = re.escape(alias)
    pre = r"\b" if alias[:1].isalnum() else ""
    post = r"\b" if alias[-1:].isalnum() else ""
    return re.compile(pre + body + post, re.IGNORECASE)


# longest alias first so "west nile fever" is consumed before "west nile"
_ALIASES = sorted(
    ((alias, name) for name, d in DISEASES.items() for alias in d["aliases"]),
    key=lambda x: len(x[0]), reverse=True)
_ALIAS_RE = [(_alias_pattern(a), name) for a, name in _ALIASES]


def scan_text(*texts):
    """Disease names mentioned anywhere in the given text.

    Takes any number of strings so the caller can pass the query, the retrieved
    chunks and the web results together. The original only ever saw the query
    and the chunks, which is why a web-routed answer never got surveillance data.

    Diseases with no aliases are absent from the ECDC extract, so they are never
    matched and never produce a data block.
    """
    blob = " ".join(t for t in texts if t)
    if not blob:
        return []
    return sorted({name for pattern, name in _ALIAS_RE if pattern.search(blob)})


class Surveillance:
    """The published CSV, plus the questions we ask of it."""

    def __init__(self, df=None, loaded_at=None, source=None):
        self.df = df
        self.loaded_at = loaded_at
        self.load_error = None
        self.source = source          # (repo_id, filename), so it can reload itself

    # --- loading ------------------------------------------------------------

    @classmethod
    def from_hub(cls, repo_id, filename):
        s = cls()
        s.load(repo_id, filename)
        return s

    def load(self, repo_id=None, filename=None):
        if repo_id is None:
            if not self.source:
                return
            repo_id, filename = self.source
        self.source = (repo_id, filename)
        try:
            # Imported here, inside the guard: a missing package is a load
            # failure like any other and must not take the process down.
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(repo_id=repo_id, filename=filename,
                                   repo_type="dataset", force_download=True)
            df = pd.read_csv(path)
            df["year"] = df["year"].astype(int)
            self.df, self.loaded_at, self.load_error = df, time.time(), None
            print(f"[ECDC] {len(df)} rows, {df['disease'].nunique()} diseases, "
                  f"{df['country'].nunique()} countries, {df['year'].min()}-{df['year'].max()}")
        except Exception as err:
            self.load_error = str(err)
            if self.df is not None:
                # A monthly refresh that fails must not throw away data that is
                # already working. The original replaced the dataframe with None
                # on any exception, so one bad reload turned a Space with good
                # surveillance data into one with none until it restarted.
                print(f"[ECDC] refresh failed, keeping the {len(self.df)} rows already "
                      f"loaded: {err}")
                return
            # Loud, because relevance gating is permissive while this is unset:
            # every disease will pass as relevant until the data comes back.
            print(f"[ECDC] LOAD FAILED, geographic gating is disabled: {err}")

    def refresh_if_stale(self):
        """Reload after 30 days, and retry a failed load with a floor on how
        often. The original had no backoff and hammered the hub on every
        request after a failure.

        Takes no arguments so the pipeline can call it without knowing where
        the data came from. It was previously called at the top of respond();
        moving the loading into this class dropped that call, and with it the
        monthly refresh, so the data would have frozen at whatever the Space
        downloaded when it last cold-started.
        """
        if not self.source:
            return
        if self.loaded_at is None:
            if self.load_error and (time.time() - (self._last_try or 0)) < RETRY_AFTER:
                return
            self._last_try = time.time()
            self.load()
        elif time.time() - self.loaded_at > REFRESH_AFTER:
            self.load()

    _last_try = None

    # --- questions ----------------------------------------------------------

    @property
    def available(self):
        return self.df is not None

    @property
    def years(self):
        if not self.available:
            return None
        return int(self.df["year"].min()), int(self.df["year"].max())

    @property
    def year_range(self):
        y = self.years
        return f"{y[0]}-{y[1]}" if y else "unknown"

    def has_country(self, country):
        """Whether the Atlas extract covers this country at all.

        Distinct from 'this country reported zero cases'. Conflating the two is
        what made three countries look disease-free.
        """
        if not self.available or not country:
            return False
        return bool((self.df["country"].str.lower() == country.lower()).any())

    def rows(self, disease, country):
        """Rows for one disease in one country, by graph name."""
        ecdc = ECDC_NAME.get(disease)
        if not self.available or not ecdc:
            return None
        sel = self.df[(self.df["disease"] == ecdc)
                      & (self.df["country"].str.lower() == country.lower())]
        return sel.sort_values("year")

    def total_cases(self, disease, country):
        r = self.rows(disease, country)
        if r is None or r.empty:
            return None
        return int(r["cases"].fillna(0).sum())

    def country_totals(self, disease):
        """Cumulative reported cases per country, best first.

        How the autochthonous and endemic country lists get checked against
        what the Atlas actually holds. Lives here so the route layer never
        touches a dataframe.
        """
        if not self.available:
            return None
        name = ECDC_NAME.get(disease, disease)
        subset = self.df[self.df["disease"] == name]
        if subset.empty:
            return None
        totals = subset.groupby("country")["cases"].sum()
        return sorted(((c, int(v)) for c, v in totals.items()), key=lambda kv: -kv[1])

    def known_diseases(self):
        return [] if not self.available else sorted(self.df["disease"].astype(str).unique())

    # --- relevance ----------------------------------------------------------

    def relevant(self, disease, country):
        """Whether this disease is worth naming for this country."""
        info = DISEASES.get(disease)
        if info is None:
            return True

        if info["gating"] == "never":
            return False
        if not country or country == "All Europe":
            return True
        if info["gating"] in ("autochthonous", "endemic"):
            return country in info.get("countries", [])

        group = info["group"]
        if group in ("1", "2b"):
            return True
        if group != "2a":
            return True

        # Group 2a with no special gating: judge on reported cases, but only
        # where there is surveillance to judge from.
        if not self.available:
            return True
        if not self.has_country(country):
            return True
        total = self.total_cases(disease, country)
        return bool(total and total > 0)

    def relevant_diseases(self, country):
        return sorted(d for d in DISEASES if self.relevant(d, country))

    # --- the context block --------------------------------------------------

    def block(self, diseases, country):
        """The surveillance text injected into the model's context.

        Empty string when there is nothing real to show. Callers must key any
        'ECDC data was used' claim off this, not off the disease list — the
        original built a source citation from the list and could claim data
        that was never injected.
        """
        if not diseases or not country or country == "All Europe":
            return ""
        if not self.available or not self.has_country(country):
            return ""

        blocks = []
        for disease in sorted(diseases):
            rows = self.rows(disease, country)
            if rows is None or rows.empty:
                continue
            info = DISEASES[disease]
            label = info["ecdc"] or disease

            cases = ", ".join(f"{int(r['year'])}: {int(r['cases'])}" for _, r in rows.iterrows())
            text = f"{label} in {country}\n  Reported cases: {cases}\n"

            if rows["deaths"].notna().any():
                deaths = ", ".join(
                    f"{int(r['year'])}: {int(r['deaths'])}" if pd.notna(r["deaths"])
                    else f"{int(r['year'])}: n/a" for _, r in rows.iterrows())
                text += f"  Deaths: {deaths}\n"

            dark = rows.iloc[0].get("dark_figure")
            if dark and pd.notna(dark):
                text += f"  Under-reporting: {dark}\n"
            note = GROUP_NOTES.get(str(rows.iloc[0].get("group", "")))
            if note:
                text += f"  {note}\n"
            blocks.append(text)

        if not blocks:
            return ""

        header = (
            f"ECDC SURVEILLANCE DATA FOR {country.upper()} ({self.year_range}):\n"
            "Source: ECDC Surveillance Atlas. Reported cases only.\n"
            "This lists only the diseases relevant to the pathways retrieved for this query, "
            f"not every disease present in {country}. Do not infer that a disease is absent "
            f"from {country} because it is not listed here. Reported case counts may include "
            "imported or travel-associated cases, so do not infer local transmission from "
            "case numbers alone.\n\n")
        return header + "\n".join(blocks)
