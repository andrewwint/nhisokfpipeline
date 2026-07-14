"""Airflow DAG — the OKF compile pipeline, runnable on aws-mwaa-local-runner.

    fetch → verify_gate → compile → slim → conformance

It is a *thin wrapper*: every task calls the same functions as `python -m okf_pipeline`. The
`verify_gate` task is the point — it runs each concept's survey-weighted analysis and **fails the
DAG if a real (non-seeded) concept is quarantined**; a quarantined concept is never compiled into
the bundle, so a confidently-wrong number physically cannot ship.

The heavy lifting lives in `okf_pipeline/` (importable on the worker); this file only orchestrates.
"""

from __future__ import annotations

import os
from pathlib import Path

import pendulum
from airflow.decorators import dag, task
from airflow.exceptions import AirflowFailException

from okf_pipeline import (
    compile_bundle,
    conformance,
    fetch,
    load,
    required_columns,
    slim,
    verify,
)
from okf_pipeline.pipeline import dump_results, load_results

# Writable scratch inside the worker/container (override with OKF_BUILD).
BUILD = Path(os.environ.get("OKF_BUILD", "/tmp/okf_build"))
BUNDLE = BUILD / "okf_bundle"
SLIM = BUILD / "microdata" / "adult23_slice.parquet"
VERIFY_JSON = BUILD / "verify.json"


@dag(
    schedule=None,                       # trigger manually for the demo
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    catchup=False,
    tags=["okf", "compile"],
    doc_md=__doc__,
)
def okf_compile():

    @task
    def fetch_csv() -> str:
        """Ensure the NHIS 2023 adult CSV is present (downloads from CDC if not)."""
        return str(fetch(BUILD / "data"))

    @task
    def verify_gate(csv: str) -> str:
        """THE GATE. Run every concept's weighted analysis; quarantine any wrong claim.
        Fail the DAG on a real regression; seeded defects are expected to quarantine."""
        results = verify(load(Path(csv)))
        for r in results:
            mark = "PASS" if r.passed else "QUARANTINE"
            print(f"{mark:11} {r.concept.id:18} claimed={r.concept.value_pct}  "
                  f"computed={r.computed_pct}  Δ={r.delta_pp}pp")
        dump_results(results, VERIFY_JSON)
        regressions = [r.concept.id for r in results if not r.passed and not r.concept.seeded_defect]
        if regressions:
            raise AirflowFailException(f"verification regression — quarantined: {regressions}")
        return str(VERIFY_JSON)

    @task
    def compile_step(verify_json: str) -> str:
        """Write only the concepts that passed the gate. Quarantined ones never become files."""
        published = compile_bundle(load_results(Path(verify_json)), BUNDLE)
        print(f"published {len(published)} verified concept(s) to {BUNDLE}")
        return str(BUNDLE)

    @task
    def slim_step(csv: str, verify_json: str) -> str:
        """duckdb projects the CSV down to only the verified columns → the deploy parquet."""
        published = [r for r in load_results(Path(verify_json)) if r.passed]
        out = slim(Path(csv), required_columns(published), SLIM)
        print(f"wrote {out} ({out.stat().st_size / 1024:,.0f} KB)")
        return str(out)

    @task
    def check_conformance(bundle: str) -> None:
        ok, issues = conformance(Path(bundle))
        if not ok:
            raise AirflowFailException(f"conformance failed: {issues}")
        print("conformance: PASS")

    csv = fetch_csv()
    verify_json = verify_gate(csv)
    bundle = compile_step(verify_json)
    slim_step(csv, verify_json)
    check_conformance(bundle)


okf_compile()
