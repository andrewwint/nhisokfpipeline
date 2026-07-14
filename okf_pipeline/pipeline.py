"""The OKF compile pipeline — six small stages, each one a DAG task.

    fetch → verify (the GATE) → compile → slim → conformance → publish

`verify` is the point: it RUNS each concept's documented, survey-weighted analysis against the
real microdata and quarantines any concept whose claimed number is wrong. A schema/link check
would pass the naive 3.66% insulin figure; only running it catches that it should be 31.96%.

Everything here is plain Python so you can run it without Airflow (`python -m okf_pipeline`); the
DAG in `dags/` is a thin wrapper that calls these same functions.
"""

from __future__ import annotations

import json
import re
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from .concepts import CONCEPTS, DESIGN_COLUMNS, Concept

# CDC NHIS 2023 Sample Adult public-use file (public domain).
NHIS_ZIP_URL = "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Datasets/NHIS/2023/adult23csv.zip"

# A concept passes the gate only if its claimed % matches the computed % within this margin.
TOLERANCE_PP = 0.5


# --- Stage 1: fetch --------------------------------------------------------------------------

def fetch(data_dir: Path = Path("data")) -> Path:
    """Ensure the NHIS 2023 adult CSV is present; download + unzip it if not.

    Set `NHIS_CSV` to reuse a copy you already have (e.g. the lab's) and skip the download.
    """
    import os

    env = os.environ.get("NHIS_CSV")
    if env and Path(env).exists():
        return Path(env)

    data_dir.mkdir(parents=True, exist_ok=True)
    csv = data_dir / "adult23.csv"
    if csv.exists():
        return csv

    zip_path = data_dir / "adult23csv.zip"
    print(f"fetch: downloading {NHIS_ZIP_URL}")
    urllib.request.urlretrieve(NHIS_ZIP_URL, zip_path)
    with zipfile.ZipFile(zip_path) as z:
        name = next(n for n in z.namelist() if n.lower().endswith(".csv"))
        z.extract(name, data_dir)
        (data_dir / name).rename(csv)
    zip_path.unlink(missing_ok=True)
    return csv


def load(csv: Path) -> pd.DataFrame:
    return pd.read_csv(csv, low_memory=False)


# --- The survey-weighted computation (what `verify` runs) ------------------------------------

def weighted_prevalence(df: pd.DataFrame, concept: Concept) -> float:
    """Survey-weighted % answering `affirmative` for `concept.variable` within its universe.

    `concept.universe` is a trusted, in-repo expression (authored here, not user input), so
    `df.query` is safe. The number is weighted by `WTFA_A` — an unweighted count would be wrong
    for NHIS. Non-response codes (anything but 1/2) are dropped from the denominator.
    """
    sub = df.query(concept.universe) if concept.universe else df
    valid = sub[sub[concept.variable].isin([1, 2])]
    denom = valid[concept.weight].sum()
    if denom == 0:
        return float("nan")
    numer = valid.loc[valid[concept.variable] == concept.affirmative, concept.weight].sum()
    return numer / denom * 100.0


# --- Stage 2: verify (THE GATE) --------------------------------------------------------------

@dataclass
class VerifyResult:
    concept: Concept
    computed_pct: float
    delta_pp: float
    verdict: str          # "PASS" | "QUARANTINE"

    @property
    def passed(self) -> bool:
        return self.verdict == "PASS"


def verify(df: pd.DataFrame, concepts: list[Concept] = CONCEPTS) -> list[VerifyResult]:
    """Run each concept's documented analysis and gate on whether the claim matches.

    This is execution-grounded verification: not "does the file parse" but "does the number the
    file claims survive being recomputed from the real data." Mismatch → QUARANTINE.
    """
    results: list[VerifyResult] = []
    for c in concepts:
        computed = weighted_prevalence(df, c)
        delta = abs(computed - c.value_pct)
        verdict = "PASS" if delta <= TOLERANCE_PP else "QUARANTINE"
        results.append(VerifyResult(c, round(computed, 2), round(delta, 2), verdict))
    return results


# --- Stage 3: compile (write only what passed) ----------------------------------------------

_FRONTMATTER = """\
---
id: {id}
title: "{title}"
variable: {variable}
analytical_universe: "{universe}"
weight: {weight}
value_pct: {value_pct}
source: "NHIS 2023 Sample Adult public-use file (adult23.csv)"
verification:
  verdict: PASS
  method: execution-grounded
  computed_pct: {computed_pct}
---
{title}. Survey-weighted (by {weight}); verified by running the documented analysis.
"""


def compile_bundle(results: list[VerifyResult], out_dir: Path) -> list[VerifyResult]:
    """Write each PASSED concept as an OKF markdown file. Quarantined concepts never become
    files — that is how "physically absent" grounding is enforced. Writes an audit `log.md`."""
    variables = out_dir / "variables"
    variables.mkdir(parents=True, exist_ok=True)

    published = [r for r in results if r.passed]
    for r in published:
        c = r.concept
        (variables / f"{c.id}.md").write_text(
            _FRONTMATTER.format(
                id=c.id, title=c.title, variable=c.variable, universe=c.universe,
                weight=c.weight, value_pct=c.value_pct, computed_pct=r.computed_pct,
            )
        )

    index = "# NHIS 2023 diabetes — verified OKF bundle\n\n"
    index += "".join(f"- [variables/{r.concept.id}](variables/{r.concept.id}.md) — {r.concept.title}\n"
                     for r in published)
    (out_dir / "index.md").write_text(index)

    log = "# Compile log\n\n| concept | claimed | computed | Δpp | verdict |\n|---|---|---|---|---|\n"
    log += "".join(
        f"| {r.concept.id} | {r.concept.value_pct} | {r.computed_pct} | {r.delta_pp} | "
        f"{'✅ PASS' if r.passed else '⛔ QUARANTINE'} |\n"
        for r in results
    )
    (out_dir / "log.md").write_text(log)
    return published


# --- Stage 4: slim (duckdb projects only the verified columns) -------------------------------

def _columns_in(expr: str) -> set[str]:
    return set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", expr))


def required_columns(published: list[VerifyResult]) -> list[str]:
    """Only the columns the PASSED concepts actually touch (+ the survey design)."""
    cols: set[str] = set(DESIGN_COLUMNS)
    for r in published:
        cols.add(r.concept.variable)
        cols |= _columns_in(r.concept.universe)
    # keep design cols + real identifiers; drop bare numbers already excluded by the regex
    return sorted(cols)


def slim(csv: Path, columns: list[str], out_path: Path) -> Path:
    """Project the big CSV down to `columns` and write the slim parquet — one streaming duckdb
    statement, no full pandas load. This is the file the deployed agent ships."""
    import duckdb

    out_path.parent.mkdir(parents=True, exist_ok=True)
    col_list = ", ".join(columns)
    duckdb.sql(
        f"COPY (SELECT {col_list} FROM read_csv_auto('{csv}')) "
        f"TO '{out_path}' (FORMAT parquet)"
    )
    return out_path


# --- Stage 5: conformance --------------------------------------------------------------------

_REQUIRED_KEYS = ("id:", "value_pct:", "verification:")


def conformance(out_dir: Path) -> tuple[bool, list[str]]:
    """Cheap structural check on the emitted bundle (the gate already did the hard part)."""
    issues: list[str] = []
    variables = out_dir / "variables"
    if not (out_dir / "index.md").exists():
        issues.append("missing index.md")
    files = list(variables.glob("*.md")) if variables.exists() else []
    if not files:
        issues.append("no verified concept files were written")
    for f in files:
        text = f.read_text()
        for key in _REQUIRED_KEYS:
            if key not in text:
                issues.append(f"{f.name}: missing '{key}'")
    return (not issues, issues)


# --- Serialization (so Airflow tasks can pass verify results between workers) ----------------

def dump_results(results: list[VerifyResult], path: Path) -> Path:
    """Persist verify results as JSON (a DataFrame can't cross an XCom; a small manifest can)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(
        [{"concept": asdict(r.concept), "computed_pct": r.computed_pct,
          "delta_pp": r.delta_pp, "verdict": r.verdict} for r in results],
        indent=2,
    ))
    return path


def load_results(path: Path) -> list[VerifyResult]:
    data = json.loads(Path(path).read_text())
    return [VerifyResult(Concept(**d["concept"]), d["computed_pct"], d["delta_pp"], d["verdict"])
            for d in data]
