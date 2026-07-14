"""The concepts the pipeline tries to publish.

A *concept* is a claim we want in the OKF bundle: a survey-weighted figure with the exact
analysis that produces it (variable, universe, weight). The pipeline's `verify` stage RUNS that
analysis against the real microdata and only publishes concepts whose claimed number matches.

Two concepts here on purpose:
  * `DIBINS_A` — the correct claim (31.96% insulin use among diagnosed diabetics), and
  * `DIBINS_A_naive` — a SEEDED DEFECT: it documents the same (diagnosed) universe but claims
    the *naive* 3.66% share. Running the documented analysis yields 31.96%, so the gate catches
    the mismatch and quarantines it. That quarantine is the whole demo.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Concept:
    id: str
    title: str
    variable: str          # the NHIS column the figure is about
    universe: str          # the row filter defining "who" (a trusted, in-repo expression)
    value_pct: float       # the CLAIMED survey-weighted percentage
    weight: str = "WTFA_A"
    affirmative: int = 1   # the "yes" code (1 = Yes for these yes/no items)
    seeded_defect: bool = False


CONCEPTS: list[Concept] = [
    Concept(
        id="DIBEV_A",
        title="Diagnosed diabetes prevalence among U.S. adults",
        variable="DIBEV_A",
        universe="",                       # empty = all adults
        value_pct=9.80,
    ),
    Concept(
        id="PREDIB_A",
        title="Prediabetes prevalence among U.S. adults",
        variable="PREDIB_A",
        universe="",
        value_pct=16.07,
    ),
    Concept(
        id="DIBINS_A",
        title="Currently takes insulin (among adults with diagnosed diabetes)",
        variable="DIBINS_A",
        universe="DIBEV_A == 1",
        value_pct=31.96,
    ),
    Concept(
        id="DIBINS_A_naive",
        title="Insulin use — SEEDED DEFECT (naive share mislabeled 'among diagnosed')",
        variable="DIBINS_A",
        universe="DIBEV_A == 1",
        value_pct=3.66,                    # the wrong (naive, whole-sample) number
        seeded_defect=True,
    ),
]

# Survey-design columns that must travel with the data for any weighted estimate + design CI.
DESIGN_COLUMNS = ["WTFA_A", "PSTRAT", "PPSU"]
