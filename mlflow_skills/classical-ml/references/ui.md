# MLflow UI Reference

Navigate the MLflow tracking server UI for classical ML workflows. The UI is the fastest way to spot regressions, compare runs, inspect datasets, and audit registry activity.

## Table of Contents
1. [Starting the UI](#1-starting-the-ui)
2. [Top-Level Layout](#2-top-level-layout)
3. [Experiments Page](#3-experiments-page)
4. [Run Detail Page](#4-run-detail-page)
5. [Compare Runs View](#5-compare-runs-view)
6. [Logged Models Page (MLflow 3)](#6-logged-models-page)
7. [Model Registry Page](#7-model-registry-page)
8. [Datasets Tab on a Run](#8-datasets-tab-on-a-run)
9. [Search Box Syntax](#9-search-box-syntax)
10. [Run Deletion / Restore](#10-run-deletion--restore)

---

## 1. Starting the UI

```bash
# Local sqlite backend
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

# Local postgres
mlflow ui --backend-store-uri postgresql://user:pw@host/db --port 5000

# Against an already-running tracking server (just shows the UI)
mlflow ui --port 5000   # defaults to the same backend as server
```

> ⚠️ MLflow 3.5+: the tracking **server** itself requires `--allowed-hosts`. If you started `mlflow server` without it, the UI may fail to load in your browser. Add `--allowed-hosts "*"` (local dev) or specific hostnames (prod).

Open `http://localhost:5000` (or your configured host:port).

---

## 2. Top-Level Layout

Sidebar (left):
- **Experiments** — list of all experiments in the backend
- **Models** — registered models + Logged Models (MLflow 3)
- **Prompts** — Prompt Registry (if you use GenAI workflows; not used by classical-ml skill)

Top bar:
- Search runs across experiments (uses `search_runs` filter syntax — §9)
- New experiment / new run buttons
- (Server-only) admin tools

---

## 3. Experiments Page

- **Columns**: Run name, Duration, Metrics (small sparkline), Params, Tags, Created
- **Sort**: click any column header
- **Filter**: type into the search bar (uses filter syntax §9)
- **Compare**: select rows via checkbox → "Compare" button → opens Compare view (§5)
- **Delete / Restore**: select rows → Delete button (soft delete)

Click an experiment name → see all runs in that experiment. Click a run → Run Detail page.

---

## 4. Run Detail Page

Tabs (order may vary by version):

| Tab | What it shows |
|---|---|
| **Overview** | Run metadata: run ID, status, start/end time, artifact URI, experiment, user, source run, parent runs |
| **Parameters** | All `log_param` values in a table |
| **Metrics** | Line charts over `step`; click to expand; filter to specific keys |
| **System Metrics** | `system/*` metrics (only if `enable_system_metrics_logging` was active) |
| **Tags** | All `set_tag` values |
| **Artifacts** | Tree view of the run's artifact directory (model files, plots, datasets) |
| **Datasets** | Datasets logged via `log_input` (lineage) |
| **Logged Models** (MLflow 3) | Quick link to the LoggedModel entities produced by this run |
| **Assessments** | Human feedback / LLM judge scores (mostly used by GenAI) |

Useful actions:
- **Copy run ID** button (top right)
- **Delete run** (top right)
- **Open in Compare** with selected runs

---

## 5. Compare Runs View

Select 2+ runs in an experiment → "Compare". The Compare view shows:

- **Parallel coordinates** plot — drag axes to filter; see which param ranges correlate with high metric
- **Scatter plot** — pick X (param/metric) and Y (metric) → see correlation
- **Box plot** — distribution of a metric grouped by a param value
- **Table** — aligned table of all params/metrics/tags across selected runs

Export: use `mlflow runs search` (CLI) to pull the same data into pandas for further analysis.

---

## 6. Logged Models Page (MLflow 3)

In MLflow 3, models are first-class entities. Sidebar → "Models" → see:
- **Registered Models** tab: registered model names, versions, aliases, tags
- **Logged Models** tab: every logged model across all experiments (by `model_id`)

Per-model actions:
- **Promote to alias** — set `champion`/`challenger` directly
- **View run** — link back to the run that produced it
- **Load code snippet** — `mlflow.pyfunc.load_model("models:/<model_id>")` ready to paste

---

## 7. Model Registry Page

Sidebar → "Models" → click a registered model name → see:
- All versions with status, aliases, tags, creation timestamp
- Per-version: source run, source experiment, description, deployment notes
- **Stage transition** (deprecated — prefer aliases)
- **Compare versions** side-by-side

Common UI tasks:
- Add/remove alias on a version
- Add tag (e.g., `validation_status: approved`)
- Edit description (Markdown)
- Delete version (soft delete)

---

## 8. Datasets Tab on a Run

Shows every dataset logged with `log_input`:
- Name
- Digest (content hash) — click to find runs trained on the same digest
- Source (URI)
- Context (`training`, `testing`, `validation`, `production`)
- Profile (row count, column types, summary stats)
- Schema (column names and types)

Click a digest to see all runs that logged the same digest — useful for "who else trained on this dataset?".

---

## 9. Search Box Syntax

The top-bar search uses the same filter syntax as `search_runs` (see `tracking.md` §7):

```text
metrics.accuracy > 0.9
metrics.accuracy > 0.9 AND params.model = "rf"
tags.`mlflow.runName` = "baseline"
attributes.status = "FINISHED"
metrics.roc_auc >= 0.85 AND tags.stage = "champion"
```

Same quoting rules as the API:
- String values in double quotes
- Keys with dots/special chars in backticks
- `OR` not supported (use AND; chain calls if you really need OR)

---

## 10. Run Deletion / Restore

- **Soft delete**: `mlflow runs delete --run-id <id>` or UI Delete button. The run is marked deleted and hidden by default; `mlflow gc` purges after retention.
- **Restore**: `mlflow runs restore --run-id <id>` or UI Restore (visible if you check "Show deleted runs").
- **Permanent purge**: `mlflow gc --backend-store-uri <uri> --older-than 30d` removes soft-deleted runs older than 30 days.

For registered models:
- Delete version: UI → Models → click version → Delete (soft)
- Delete whole registered model: UI → Models → click model → top-right Delete
- `mlflow gc` purges soft-deleted registry entries too

---

## See also

- `tracking.md` §9 for the CLI equivalents of every UI action
- `registry.md` §9 for registry CLI
- `troubleshooting.md` for "I can't find this run" debugging