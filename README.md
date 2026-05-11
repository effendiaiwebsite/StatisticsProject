# Hybrid Outlier Detection

A small FastAPI + Jinja web app that runs **four** outlier detectors on the
same column (or matrix), then takes a **majority-style vote**:

| Votes against a row | Verdict           |
|--------------------:|-------------------|
| 0                   | Normal            |
| 1                   | Possible          |
| >=2                 | High-confidence   |

The four detectors:
1. **Modified Z-Score** (Iglewicz & Hoaglin) using MAD, with a mean-abs-dev
   fallback when MAD = 0.
2. **IQR fences** with multiplier `k`.
3. **Isolation Forest** (scikit-learn) with a user-set contamination.
4. **Iterative mean/std** from the provided PDF: at iteration `i`, drop
   points farther than `f_i * sigma` from the current mean, where
   `f_i = (I - i) / 10`.

Formal pseudocode for every detector and the vote lives in
[docs/algorithm.md](docs/algorithm.md).

---

## Setup

```bash
python -m venv .venv
. .venv/Scripts/activate          # PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py
```

Then open **http://127.0.0.1:8000** in a browser. The app must be served
over HTTP (do **not** double-click `templates/index.html` — Jinja, the API,
and the same-origin assumption will all break).

Set the `PORT` environment variable to bind a different port:

```bash
PORT=8080 python main.py
```

---

## Deploy on Render

`render.yaml` is committed. From the Render dashboard, choose
**New → Blueprint** and point it at this repo; Render reads `render.yaml`
and creates a free web service with:

- `pip install -r requirements.txt` as the build command
- `python main.py` as the start command (reads `$PORT` automatically)

No other configuration is needed.

---

## What each knob actually controls

The three knobs are **not** all-encompassing — each one only steers the
detectors it makes sense for. The remaining detectors keep their defaults.

### Strictness (only affects MAD and IQR)

A single pill picker (Lenient / Balanced / Strict) sets two parameters:

| Preset    | `tau` (MAD threshold) | `k` (IQR multiplier) |
|-----------|----------------------:|---------------------:|
| Lenient   | 4.0                   | 2.0                  |
| Balanced  | 3.5 *(default)*       | 1.5                  |
| Strict    | 3.0                   | 1.0                  |

- Higher `tau` -> the Modified Z must be further from zero before MAD flags
  a point. Lenient lets more values through; Strict flags more.
- Higher `k` -> the IQR fences are wider. Same direction: Lenient lets more
  through.

Strictness does **nothing** to Isolation Forest or the Iterative method.

### Contamination (only affects Isolation Forest)

A slider from `0.01` to `0.20` (i.e. 1%..20%). This is the fraction of the
data the Isolation Forest model expects to be outliers, and it directly
controls how it sets its decision threshold. Larger contamination -> more
points flagged, regardless of how the rest of the data looks.

Contamination does **nothing** to MAD, IQR, or the Iterative method.

### Iterations (only affects the Iterative mean/std method)

A slider from `1` to `100`. `I` controls both how long the algorithm runs
**and** the threshold schedule, because the cutoff at step `i` is
`f_i = (I - i) / 10` sigmas. Two important consequences:

1. The method is designed to converge. With enough iterations it will
   eventually drop nearly every point — `f_i` falls below `1.5`, then
   below `1.0`, then below `0.5`, etc.
2. To stop the iterative method from dominating the vote, only points
   removed while `f_i >= 1.5` are vote-eligible. The UI shows both
   `total_removed` and `flagged_for_vote`.

The convergence chart on the results page also auto-detects the iteration
`K` where sigma stops dropping meaningfully (relative drop < 1%) and tells
you whether you are under-, correctly-, or over-iterating.

Iterations does **nothing** to MAD, IQR, or Isolation Forest.

---

## Three-page UI flow

1. **Home** — pick Univariate or Multivariate (and grab the sample files).
2. **Upload & configure** — drag-drop CSV/XLSX, preview the first 10 rows,
   pick the numeric column(s), set strictness / contamination / iterations,
   choose whether the cleaned download should *remove* or *flag* outliers.
3. **Results** — summary cards, a Plotly chart (boxplot+scatter for 1-D, a
   PCA-2D projection + per-column boxplots for N-D), a convergence chart,
   per-detector explanations, a "why was this row flagged" list, a
   sortable / filterable results table, and CSV / JSON downloads. The
   live-tweak panel at the top re-runs the analysis on the fly
   (~350 ms slider debounce; pills fire immediately).

---

## Sample data

Two CSVs ship in `sample_data/`:

- `univariate_demo.csv` — a stable `~N(100, 0.3)` measurement with several
  obvious spike outliers injected (rows 51, 57, 74, 86). The detectors
  should agree on these.
- `multivariate_demo.csv` — synthetic `(height_cm, weight_kg, resting_hr)`
  trios where height ~ weight in roughly linear fashion. A few rows are
  outliers in *combination* (e.g. very heavy for their height, or very tall
  for the rest of the population): try rows 50, 55, 71, 85.

You can grab them from the Home page or directly at
`/sample/univariate_demo.csv` and `/sample/multivariate_demo.csv`.

---

## Session handling

Uploaded files are kept in an **in-memory `dict` keyed by a UUID**
session ID. This is fine for a single-instance demo and for Render's free
plan, but uploads are lost on process restart and on horizontally-scaled
deployments because the session won't exist on every replica. For
production, back the store with disk (`pickle` per session) or with an
object store like S3, and set up a TTL sweeper. The `/upload` route is the
only place that creates a session; `/analyze`, `/download/csv`, and
`/download/report/{id}` read from it.

---

## Robustness notes

- Frontend renders defensively: a missing stats key simply skips that
  card rather than crashing the page.
- Iterations are clipped server-side to `[1, 100]` regardless of slider.
- Contamination is clipped server-side to `[0.01, 0.20]`.
- An upload with no numeric columns returns a clean 400 error.
- An analysis with fewer than 3 non-null numeric values returns a clean
  400 error rather than silently producing nonsense.
- CORS is not configured because the app is same-origin. Open it via
  `http://127.0.0.1:8000`, not by double-clicking the HTML file.
