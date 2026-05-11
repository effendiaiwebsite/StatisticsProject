# Hybrid Outlier Detection

**FSCT 7910 — NET: Research Methodology and Measurement Models**
British Columbia Institute of Technology

**Team:** Muskan Walia · Saif Chhipa · Hanson Ho · Satinder Sandhu

---

## 1. Overview

This project investigates the practical problem of identifying outliers in
small-to-medium numeric datasets when no single detector is universally
appropriate. We implement four well-known outlier detectors — three
classical statistical procedures and one tree-based machine-learning
method — and combine their decisions into a transparent **voting
ensemble**. The result is an interactive web application that uploads a
CSV/XLSX file, runs all four detectors, and presents a side-by-side
comparison with live parameter tuning.

The motivating question is methodological:

> *When detectors disagree, what should an analyst conclude — and how
> should the tradeoffs they encode (robustness, threshold choice, model
> assumptions) be exposed to the user rather than hidden in defaults?*

Our answer is a three-class verdict (**Normal / Possible / High-confidence**)
derived from a simple majority vote, paired with transparent controls
that map directly onto specific detector parameters.

---

## 2. Background

A single outlier procedure embeds non-trivial assumptions about the data:

- **Modified Z-Score (MAD)** assumes approximate unimodality and uses
  the median absolute deviation as a robust scale estimator. It tolerates
  ~50% contamination but degenerates when MAD = 0.
- **Tukey's IQR fences** are distribution-free in the sense that they
  use only quartile information, but the fence multiplier `k` is a tuning
  choice with no canonical optimum.
- **Isolation Forest** (Liu, Ting & Zhou, 2008) is non-parametric and
  handles multivariate interactions natively, but it requires the
  analyst to specify the expected contamination fraction.
- **Iterative mean/std trimming** (provided in the FSCT 7910 reference
  PDF) is intuitive and visualisable, but converges by design — it will
  eventually remove the bulk of the data if run long enough, which makes
  it sensitive to the iteration count `I`.

No procedure is "right" in isolation. The contribution of this project is
to make the disagreement *visible* and to let the user explore it
interactively.

---

## 3. Methods

### 3.1 Modified Z-Score (MAD)

For a sample `x = (x_1, ..., x_n)` we compute the Modified Z-Score of
Iglewicz & Hoaglin (1993):

```
M_i = 0.6745 · (x_i − median) / MAD
MAD = median(|x_i − median|)
```

The factor `0.6745 = 1/Φ⁻¹(0.75)` makes MAD a consistent estimator of
σ under normality, so `|M_i|` is on the same scale as a standard
z-score. A point is flagged when `|M_i| > τ`.

When `MAD = 0` (e.g. when more than half of the values are identical),
we fall back to a mean-absolute-deviation estimator scaled by
`1.2533 = √(π/2)`.

### 3.2 IQR fences

Following Tukey, we compute the first and third quartiles and flag any
point lying outside

```
[Q1 − k · IQR, Q3 + k · IQR],   IQR = Q3 − Q1
```

This procedure is robust because quartiles are not displaced by extreme
values, but it relies on the analyst to pick `k`.

### 3.3 Isolation Forest

We use scikit-learn's `IsolationForest` with `n_estimators = 200`. The
score for a point `x` is

```
s(x) = 2^( − E(h(x)) / c(n) )
```

where `E(h(x))` is the average path length to isolate `x` across the
forest and `c(n)` is the expected path length for `n` samples in an
unsuccessful binary-search-tree query. Points isolated more easily
receive higher scores. The contamination parameter sets the score
quantile at which the model declares an outlier.

### 3.4 Iterative mean / std (PDF reference method)

Starting from `μ_0 = mean(x), σ_0 = std(x)`, we iterate `i = 1, ..., I`:

```
f_i = (I − i) / 10                     # threshold in sigmas
S_i = { x ∈ S_{i−1} : |x − μ_{i−1}| ≤ f_i · σ_{i−1} }
μ_i = mean(S_i),  σ_i = std(S_i)
```

A critical observation is that `I` controls **both** the run length and
the threshold schedule, because the cutoff at step `i` is itself a
function of `I`. The schedule for `I = 20` runs from `f_1 = 1.9 σ` down
to `f_{20} = 0`, so the method is mathematically guaranteed to remove
essentially every point given enough iterations.

#### Vote-eligibility nuance

A naïve voting scheme that counted every iterative removal would let
this detector dominate the ensemble whenever `I` is large. We therefore
distinguish two quantities:

- `total_removed` — every point the loop ever dropped;
- `flagged_for_vote` — only those removed while `f_i ≥ 1.5`.

The 1.5 σ threshold corresponds roughly to where MAD and IQR start to
disagree with each other (Strict: 1.0, Balanced: 1.5, Lenient: 2.0), and
serves as a principled cutoff for vote-eligibility. Both numbers are
surfaced in the UI, along with an auto-detected "go-flat" iteration `K`
defined as the first `i` for which the relative drop in `σ` is below 1%.

### 3.5 Voting ensemble

Each of the four detectors independently produces a binary flag per row.
The vote count `c ∈ {0, 1, 2, 3, 4}` is the sum of flags, and the
verdict is

```
c = 0          →  Normal
c = 1          →  Possible
c ≥ 2          →  High-confidence
```

For multivariate inputs, MAD / IQR / Iterative are applied per column
and a row is flagged for the method if *any* column flags it (OR
aggregation). Isolation Forest is run natively on the full numeric
matrix and so picks up combination outliers that no per-column method
can.

Formal pseudocode for all four detectors, the vote, and the multivariate
aggregator follows the *Writing Algorithms — Samples* style and lives in
[docs/algorithm.md](docs/algorithm.md).

---

## 4. Parameter controls

We deliberately resist a single "sensitivity" slider, because the four
detectors are sensitive to different things. Three independent controls
are exposed:

| Control                 | Detector(s) it affects          | What it does                                                                 |
|-------------------------|---------------------------------|------------------------------------------------------------------------------|
| **Strictness** (pill)   | MAD, IQR                        | Sets `τ` (MAD) and `k` (IQR) together. Lenient `(4.0, 2.0)`; Balanced `(3.5, 1.5)`; Strict `(3.0, 1.0)`. |
| **Contamination** (slider) | Isolation Forest             | Expected outlier fraction in `[0.01, 0.20]`. Sets the score quantile.        |
| **Iterations** (slider) | Iterative mean / std            | `I ∈ [1, 100]`. Controls both run length and the threshold schedule.        |

Each control has a `?` icon in the UI that explains the underlying
formulas. Crucially, the three controls are *not* redundant: changing
strictness does nothing to Isolation Forest, and changing contamination
does nothing to MAD/IQR. This separation is deliberate and is part of
the methodological argument of the project.

---

## 5. Implementation

### Stack

- **Backend.** Python 3.11+, FastAPI, pandas, numpy, scipy, scikit-learn,
  Jinja2. Sessions are stored in an in-memory dictionary keyed by UUID;
  this is fine for a single-process demo but would need to be backed by
  disk or an object store for production multi-instance deployment.
- **Frontend.** A single Jinja-rendered HTML page using Tailwind (via
  CDN), Plotly.js for visualisations, and vanilla JavaScript. No build
  step is required.

### Routes

- `GET  /`                      — application UI
- `POST /upload`                — accepts a CSV/XLSX, creates a session
- `POST /load-sample`           — loads a bundled sample by filename
- `POST /analyze`               — runs all four detectors and the vote
- `POST /download/csv`          — cleaned CSV (flag or remove mode)
- `GET  /download/report/{id}`  — full JSON report
- `GET  /sample/{name}`         — raw sample CSV download
- `GET  /healthz`               — health check

### Visualisations

The results view contains three Plotly figures:

1. A distribution plot (univariate boxplot + per-row scatter, or a
   2-D PCA projection plus per-column boxplots in the multivariate
   case).
2. An iterative-convergence chart mirroring the "Mean and StdDev over
   iterations" figure from the reference PDF — `μ` on the left axis
   and `σ` on the right axis, with a dashed vertical line at the
   auto-detected go-flat iteration `K`.
3. A "live-tweak" panel pinned to the top of the results page that
   re-runs the analysis on a ~350 ms debounce when any slider moves
   (immediately when the strictness pill is changed).

---

## 6. Sample datasets

Two synthetic datasets with known ground truth are bundled to let the
reader sanity-check the detectors. Four additional course samples are
included as `sample_1.csv` through `sample_4.csv`.

| File                       | Mode         | What it tests                                                                       |
|----------------------------|--------------|--------------------------------------------------------------------------------------|
| `univariate_demo.csv`      | Univariate   | `~ N(100, 0.3)` baseline with four injected spikes at rows 51, 57, 74, 86.           |
| `multivariate_demo.csv`    | Multivariate | `(height, weight, resting_hr)` triples; rows 50, 71, 85 are obvious outliers, row 55 is a *combination* outlier (weight 55 kg with height 176 cm) that only Isolation Forest reliably picks up. |
| `sample_1.csv` … `sample_4.csv` | Univariate | Course-supplied data; expected to be analysed under varying strictness/iterations to demonstrate detector behaviour. |

On the univariate demo at the Balanced preset with `I = 20`, all four
injected spikes verdict as **High-confidence** with 4/4 votes; on the
multivariate demo, row 55 verdicts as **Possible** (1/4 — only IsoForest
catches it), illustrating exactly the kind of disagreement the ensemble
is designed to surface.

---

## 7. Running locally

```bash
python -m venv .venv
. .venv/Scripts/activate           # PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py
```

Then open **http://127.0.0.1:8000** in a browser. The application must
be served over HTTP — opening `templates/index.html` directly will not
work because Jinja templating, the FastAPI endpoints, and the
same-origin assumption all depend on a live server.

The `PORT` environment variable overrides the default of 8000.

## 8. Deployment

A Render Blueprint (`render.yaml`) is included. From the Render
dashboard, choose **New → Blueprint** and select this repository; Render
reads `render.yaml` and provisions a single Web Service that runs
`python main.py` under the Python 3.11.9 runtime. The application reads
`$PORT` automatically.

On Render's free tier the service sleeps after 15 minutes of inactivity
and incurs a ~30–60 second cold start on the next request. Any uploaded
session is held in memory and is therefore lost on restart; this is an
acceptable limitation for a demonstration but would need to be addressed
for production use.

---

## 9. Discussion

A few observations from building and stress-testing the application:

1. **Single-knob abstractions are misleading.** Early designs collapsed
   strictness, contamination, and iteration count into a single "how
   sensitive should I be?" slider. This obscured the fact that the four
   detectors have qualitatively different failure modes — IsoForest can
   silently flag the wrong fraction of the data, and the iterative
   method's threshold schedule is non-monotonic in `I` for any fixed
   real-time budget. Keeping the three controls separate, with `?` hints
   that explain *which detector each one moves*, restored that
   transparency.
2. **The iterative method needs the vote-eligibility rule.** Without the
   `f_i ≥ 1.5` cutoff, the iterative detector flags everything at high
   `I` and effectively becomes a one-detector ensemble. Surfacing both
   `total_removed` and `flagged_for_vote` made this trade-off legible.
3. **PCA is enough for a sanity check in multivariate mode.** A 2-D
   PCA projection of the standardised matrix is not a substitute for a
   formal multivariate test, but it gives the analyst a quick visual
   confirmation that the rows IsoForest flags really do sit away from
   the dense cluster.

### Limitations and future work

- Sessions are in-memory; multi-instance scale-out requires a shared
  store.
- The iterative method assumes a single mode; on bimodal data both
  modes' tails get clipped symmetrically, which is rarely what one
  wants. A mixture-aware variant would be a natural extension.
- The ensemble currently treats all four votes equally. A weighted
  vote — weighting IsoForest higher in genuinely multivariate problems
  and the classical detectors higher in univariate ones — is a
  reasonable next experiment.

---

## 10. References

- Iglewicz, B. & Hoaglin, D. C. (1993). *How to Detect and Handle
  Outliers.* ASQC Quality Press.
- Tukey, J. W. (1977). *Exploratory Data Analysis.* Addison-Wesley.
- Liu, F. T., Ting, K. M., & Zhou, Z.-H. (2008). *Isolation Forest.*
  Eighth IEEE International Conference on Data Mining.
- FSCT 7910 reference PDF, *Algorithm Design — Iterative outlier
  trimming.*
- *Writing Algorithms — Samples* (style guide used for the pseudocode
  in [docs/algorithm.md](docs/algorithm.md)).

---

## Project structure

```
main.py                — FastAPI backend (four detectors, voting ensemble, sessions, routes)
templates/index.html   — single Jinja-rendered SPA (Tailwind + Plotly + vanilla JS)
docs/algorithm.md      — formal pseudocode for all detectors and the vote
sample_data/           — bundled CSVs
requirements.txt       — pinned Python dependencies
render.yaml            — Render Blueprint
```
