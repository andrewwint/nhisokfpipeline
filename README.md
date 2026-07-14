# nhisokfpipeline — the OKF compile pipeline

The **build step for knowledge.** This pipeline turns messy CDC NHIS microdata into a **verified
OKF bundle** — the same bundle the grounded agent in
[nhisokfchat](https://github.com/andrewwint/nhisokfchat) deploys. It's the companion to article 2
(the pipeline side); [nhisokfchat](https://github.com/andrewwint/nhisokfchat) is the agent side.

Six small stages, one of which is the whole point:

```
fetch → verify (THE GATE) → compile → slim → conformance
```

**`verify` is execution-grounded, not a lint.** It *runs* each concept's survey-weighted analysis
against the real data and **quarantines any concept whose claimed number is wrong** — so a
confidently-wrong figure physically never becomes a file, and the agent can never serve it.

## Quick start (no Airflow)

```bash
python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt
# point at a CSV you have, or omit NHIS_CSV to download it from CDC:
NHIS_CSV=../nhis-okf-compiler/data/adult23.csv ./.venv/bin/python -m okf_pipeline
```

You'll watch the gate quarantine a seeded defect:

```
② verify (the gate — runs each concept's weighted analysis)
   ✅ PASS       DIBEV_A         claimed=  9.8%  computed=  9.8%   Δ=0.0pp
   ✅ PASS       PREDIB_A        claimed=16.07%  computed=16.07%   Δ=0.0pp
   ✅ PASS       DIBINS_A        claimed=31.96%  computed=31.96%   Δ=0.0pp
   ⛔ QUARANTINE DIBINS_A_naive  claimed= 3.66%  computed=31.96%   Δ=28.3pp
③ compile → wrote 3 verified concept(s)   (the 3.66% one is never written)
④ slim (duckdb) → build/microdata/adult23_slice.parquet
⑤ conformance PASS
✅ every seeded defect was quarantined; only verified figures shipped.
```

The output under `build/` — `okf_bundle/` (verified markdown) + `microdata/adult23_slice.parquet`
— is exactly what the deployed agent ships.

## The star beat: verification-as-a-gate

`concepts.py` defines a real claim (`DIBINS_A` = 31.96% insulin use among diagnosed diabetics) and
a **seeded defect** (`DIBINS_A_naive`) that claims the *naive* 3.66% for the same universe. A
schema/link check would pass the 3.66% — it's structurally clean. Only **running the weighted
analysis** reveals it should be 31.96%, so the gate quarantines it. That is the entire reason a
verification pipeline exists instead of a folder of markdown.

## Layout

```
nhisokfpipeline/
├── notebooks/01_explore_adult23.py   # §2 — the origin story (find the skip-pattern + the gotcha)
├── okf_pipeline/                     # the six stages, as plain testable functions
│   ├── concepts.py                   #   the claims (incl. the seeded defect)
│   ├── pipeline.py                   #   weighted_prevalence + fetch/verify/compile/slim/conformance
│   └── __main__.py                   #   run it all: python -m okf_pipeline
├── dags/okf_compile_dag.py           # a THIN Airflow DAG over the same functions
├── requirements.txt
└── README.md
```

The DAG is deliberately thin — it imports `okf_pipeline` and wires the functions as tasks. The
same code runs whether you invoke it as a script or as an Airflow DAG.

## Run it on Airflow (aws-mwaa-local-runner)

The DAG is written for [aws-mwaa-local-runner](https://github.com/aws/aws-mwaa-local-runner) — a
local, Docker-based Airflow that matches Amazon MWAA. (Heads up: this is **Docker-heavy** to stand
up; the Quick Start above needs none of it.)

1. Clone `aws-mwaa-local-runner`.
2. Copy this repo's `dags/okf_compile_dag.py` **and** the `okf_pipeline/` package into the runner's
   `dags/` folder (so the worker can import `okf_pipeline`).
3. Add `pandas`, `duckdb`, `pyarrow` to the runner's `requirements/requirements.txt`.
4. `./mwaa-local-env build-image && ./mwaa-local-env start`, open the Airflow UI, and trigger the
   `okf_compile` DAG. Watch the `verify_gate` task log quarantine `DIBINS_A_naive`.

## Scale-up (named, not built)

When verification is too heavy for an Airflow worker (large microdata, bootstrap CIs, many
concepts), the `verify_gate` task submits a **SageMaker Processing** job instead of computing
in-worker — same code, bigger box. That's the road not taken here, on purpose.

## Where this comes from

The verified bundle produced here is what
[nhisokfchat](https://github.com/andrewwint/nhisokfchat) deploys as a grounded agent (article 1).
The fuller statistical engine — design-based CIs, more concepts, cross-year trends — lives in the
lab, [nhis-okf-compiler](https://github.com/andrewwint/nhis-okf-compiler). This repo is the
minimal, runnable *pipeline* the article walks through.
