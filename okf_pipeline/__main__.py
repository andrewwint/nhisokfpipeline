"""Run the whole pipeline without Airflow:  python -m okf_pipeline

    fetch → verify (gate) → compile → slim → conformance

Prints the verify table (so you can watch the seeded 3.66% concept get quarantined) and writes
the verified bundle + slim parquet under ./build/.
"""

from __future__ import annotations

from pathlib import Path

from . import (
    CONCEPTS,
    compile_bundle,
    conformance,
    fetch,
    load,
    required_columns,
    slim,
    verify,
)

BUILD = Path("build")
BUNDLE = BUILD / "okf_bundle"
SLIM = BUILD / "microdata" / "adult23_slice.parquet"


def main() -> int:
    print("① fetch")
    csv = fetch()
    df = load(csv)
    print(f"   {csv}  ({len(df):,} rows)\n")

    print("② verify (the gate — runs each concept's weighted analysis)")
    results = verify(df, CONCEPTS)
    for r in results:
        mark = "✅ PASS      " if r.passed else "⛔ QUARANTINE"
        print(f"   {mark} {r.concept.id:18} claimed={r.concept.value_pct:>6}%  "
              f"computed={r.computed_pct:>6}%  Δ={r.delta_pp:>5}pp")
    quarantined = [r for r in results if not r.passed]
    print(f"   → {len(quarantined)} concept(s) quarantined; they will NOT be published.\n")

    print("③ compile (write only what passed)")
    published = compile_bundle(results, BUNDLE)
    print(f"   wrote {len(published)} verified concept(s) to {BUNDLE}/variables/\n")

    print("④ slim (duckdb → the deploy parquet)")
    cols = required_columns(published)
    slim(csv, cols, SLIM)
    size_kb = SLIM.stat().st_size / 1024
    print(f"   {SLIM}  ({size_kb:,.0f} KB, columns: {', '.join(cols)})\n")

    print("⑤ conformance")
    ok, issues = conformance(BUNDLE)
    print(f"   {'PASS' if ok else 'FAIL'}" + ("" if ok else f" — {issues}"))

    # The invariant the demo asserts: a seeded defect must never reach the bundle.
    published_ids = {r.concept.id for r in published}
    leaked = [r.concept.id for r in results if r.concept.seeded_defect and r.concept.id in published_ids]
    if leaked:
        print(f"\n❌ INVARIANT VIOLATED: a seeded defect was published: {leaked}")
        return 1
    print("\n✅ every seeded defect was quarantined; only verified figures shipped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
