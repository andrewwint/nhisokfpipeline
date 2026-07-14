# %% [markdown]
# # 01 — Exploring `adult23.csv`: the origin story
#
# **Article 2, §2.** Before there is a pipeline, there is a notebook. This is where we *look at
# the data* and discover the verification logic — the skip-pattern, the survey weight, and the
# one number that is confidently wrong if you compute it naively.
#
# Nothing here is production. The point is the opposite: these cells are the **sketch** that we
# later *extract* into a registry, a verifier, and a DAG (§3–§6). Run it top to bottom.
#
# > Open as a notebook in VS Code / Jupyter (the `# %%` markers are cells), or convert with
# > `jupytext --to ipynb 01_explore_adult23.py`.

# %%
from pathlib import Path

import pandas as pd

pd.set_option("display.max_columns", 30)

# The public-use file lives in the lab today (the pipeline will `fetch` it in §4). Point at the
# lab's copy, or set NHIS_CSV to your own.
import os

CANDIDATES = [
    os.environ.get("NHIS_CSV"),
    "../../nhis-okf-compiler/data/adult23.csv",
    "../nhis-okf-compiler/data/adult23.csv",
    "data/adult23.csv",
]
CSV = next((p for p in CANDIDATES if p and Path(p).exists()), None)
assert CSV, "Set NHIS_CSV to the NHIS 2023 adult public-use CSV (adult23.csv)."
print("using:", CSV)

# %% [markdown]
# ## 1. Load and look
#
# ~29 MB, ~29k sample adults, ~600 columns. We only care about a handful — but first, look.

# %%
df = pd.read_csv(CSV, low_memory=False)
print(df.shape)
df[["DIBEV_A", "DIBINS_A", "PREDIB_A", "DIBAGETC_A", "SEX_A", "WTFA_A", "PSTRAT", "PPSU"]].head()

# %% [markdown]
# ## 2. The skip-pattern — `DIBINS_A` isn't asked of everyone
#
# "Currently takes insulin" (`DIBINS_A`) is a **skip-pattern** item: it's only asked of adults
# who were told they have diabetes *or* prediabetes. For everyone else it's not-in-universe.
# That is the trap: if you treat "not asked" as "not taking insulin," you deflate the number.
#
# Valid responses are `1 = Yes`, `2 = No`. Look at where a valid response even exists:

# %%
asked = df["DIBINS_A"].isin([1, 2])
universe = (df["DIBEV_A"] == 1) | (df["PREDIB_A"] == 1)
print(f"valid DIBINS_A responses:            {asked.sum():>6,}")
print(f"adults with diabetes OR prediabetes: {universe.sum():>6,}")
print(f"asked ⊆ (diabetes|prediabetes)?      {bool((asked & ~universe).sum() == 0)}")
# → DIBINS_A is answered essentially only within (DIBEV_A == 1) | (PREDIB_A == 1).

# %% [markdown]
# ## 3. The gotcha — naive vs. survey-weighted
#
# The claim we want to publish is **"% of diagnosed diabetics currently taking insulin."** Two
# ways to compute it; only one is right.

# %%
# ❌ Naive: count "yes" over the WHOLE sample (unweighted). The un-asked silently become "no".
naive = (df["DIBINS_A"] == 1).sum() / len(df) * 100

# ✅ Correct: among DIAGNOSED adults (DIBEV_A == 1), survey-weighted by WTFA_A.
diag = df[df["DIBEV_A"] == 1]
num = diag.loc[diag["DIBINS_A"] == 1, "WTFA_A"].sum()
den = diag.loc[diag["DIBINS_A"].isin([1, 2]), "WTFA_A"].sum()
weighted = num / den * 100

print(f"❌ naive insulin share (whole sample):        {naive:5.2f}%")
print(f"✅ weighted, among diagnosed (DIBEV_A == 1):  {weighted:5.2f}%")
print(f"   → off by ~{weighted / naive:.0f}×.  This gap is the whole reason the pipeline exists.")
# Expected (the lab's verified result): naive ≈ 3.66%, weighted ≈ 31.96%.

# %% [markdown]
# **This is the concept the verifier must gate on.** A schema/link check passes the naive 3.66%
# — it's structurally clean. Only *running the weighted analysis* catches that it's wrong. That
# catch is §5's star beat.

# %% [markdown]
# ## 4. The survey design — nothing here is a simple average
#
# Every figure is weighted by `WTFA_A`, and its confidence interval is **design-based** over the
# survey's strata (`PSTRAT`) and PSUs (`PPSU`). These three columns must travel with the data.

# %%
df[["WTFA_A", "PSTRAT", "PPSU"]].describe().loc[["count", "min", "max"]]

# %% [markdown]
# ## 5. What we just learned → what to extract into the compiler (§3)
#
# The notebook discovered exactly what the production classes need to encode:
#
# | Notebook finding | Extracted into |
# | --- | --- |
# | `DIBINS_A` universe = `(DIBEV_A==1) | (PREDIB_A==1)`; clinical denom = `DIBEV_A==1` | **registry** (per-variable universe, weight, valid codes) |
# | weight = `WTFA_A`, design = `PSTRAT`/`PPSU` | **registry** + the survey-stats engine |
# | claim: "31.96% among diagnosed" | a **concept** to publish |
# | naive 3.66% is wrong | the **execution-grounded verifier** (run it, check it, quarantine it) |
#
# Those become `registry.py`, `concepts/*.yaml`, and `verify.py` — the compile step the DAG runs.

# %% [markdown]
# ## 6. Preview — slim the data for deploy with duckdb (§6)
#
# Once the concepts pass the gate, we know the *only* columns they touch. Project the 29 MB CSV
# down to those (+ the design columns) and write the ~314 KB parquet the agent ships — one
# streaming SQL statement, no full pandas load.

# %%
import duckdb

SLIM = "adult23_slice.parquet"
duckdb.sql(
    f"""
    COPY (
        SELECT DIBEV_A, DIBINS_A, PREDIB_A, DIBAGETC_A, SEX_A, WTFA_A, PSTRAT, PPSU
        FROM read_csv_auto('{CSV}')
    ) TO '{SLIM}' (FORMAT parquet)
    """
)
size_kb = Path(SLIM).stat().st_size / 1024
print(f"wrote {SLIM}: {size_kb:,.0f} KB  ({df.shape[0]:,} rows, 8 columns)")
# This slim parquet is exactly the file article 1's `tool_analyze_rows` queries in the deploy.

# %% [markdown]
# ## Next: cells → classes → DAG
#
# - §3 — extract these findings into `registry.py` / `concepts/*.yaml` / `verify.py`.
# - §4 — wire `fetch → verify(gate) → compile → slim(duckdb) → conformance → publish` as an
#   Airflow DAG, runnable on `aws-mwaa-local-runner`.
# - §5 — watch the `verify` task **quarantine** the naive 3.66% concept before it can publish.
#
# The notebook is the sketch. The pipeline is the compiler.
