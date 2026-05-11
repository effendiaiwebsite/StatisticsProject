"""Hybrid Outlier Detection web app.

Backend implements four detectors:
  1. Modified Z-Score (MAD)
  2. IQR fences
  3. Isolation Forest
  4. Iterative mean/std (PDF 1 method)

Voting ensemble: 0=Normal, 1=Possible, >=2=High-confidence.

Run: `python main.py` (binds 0.0.0.0:$PORT, falls back to 8000).
Open via http://127.0.0.1:8000 — do NOT double-click the HTML.
"""
from __future__ import annotations

import io
import json
import math
import os
import uuid
from typing import Any

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

app = FastAPI(title="Hybrid Outlier Detection")
templates = Jinja2Templates(directory="templates")

# In-memory session store. Single-instance demo only; lost on restart.
SESSIONS: dict[str, dict[str, Any]] = {}

STRICTNESS_PRESETS = {
    "lenient":  {"tau": 4.0, "k": 2.0},
    "balanced": {"tau": 3.5, "k": 1.5},
    "strict":   {"tau": 3.0, "k": 1.0},
}

# --------------------------------------------------------------------------
# Detectors
# --------------------------------------------------------------------------

def detect_mad(values: np.ndarray, tau: float) -> dict[str, Any]:
    """Modified Z-Score using MAD; fall back to mean abs deviation when MAD=0."""
    values = np.asarray(values, dtype=float)
    median = float(np.median(values))
    abs_dev = np.abs(values - median)
    mad = float(np.median(abs_dev))
    fallback_used = False
    if mad == 0.0:
        # Fallback: mean absolute deviation, scaled by 1.2533 for normality
        mean_abs_dev = float(np.mean(abs_dev))
        fallback_used = True
        if mean_abs_dev == 0.0:
            scores = np.zeros_like(values)
        else:
            scores = 0.6745 * (values - median) / (1.2533 * mean_abs_dev)
    else:
        scores = 0.6745 * (values - median) / mad
    flags = np.abs(scores) > tau
    return {
        "scores": scores.tolist(),
        "flags": flags.tolist(),
        "stats": {
            "median": median,
            "mad": mad,
            "tau": tau,
            "fallback_used": fallback_used,
            "flagged": int(flags.sum()),
        },
    }


def detect_iqr(values: np.ndarray, k: float) -> dict[str, Any]:
    """Standard IQR fence test."""
    values = np.asarray(values, dtype=float)
    q1 = float(np.percentile(values, 25))
    q3 = float(np.percentile(values, 75))
    iqr = q3 - q1
    lo = q1 - k * iqr
    hi = q3 + k * iqr
    flags = (values < lo) | (values > hi)
    scores = np.where(values < lo, (lo - values) / max(iqr, 1e-12),
                      np.where(values > hi, (values - hi) / max(iqr, 1e-12), 0.0))
    return {
        "scores": scores.tolist(),
        "flags": flags.tolist(),
        "stats": {
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
            "k": k,
            "lower_fence": lo,
            "upper_fence": hi,
            "flagged": int(flags.sum()),
        },
    }


def detect_isoforest(matrix: np.ndarray, contamination: float, seed: int = 42) -> dict[str, Any]:
    """Isolation Forest on the numeric matrix (n_samples x n_features)."""
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    contamination = float(np.clip(contamination, 0.01, 0.20))
    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=seed,
    )
    model.fit(matrix)
    raw = model.decision_function(matrix)  # higher = more normal
    preds = model.predict(matrix)          # -1 outlier, 1 inlier
    flags = (preds == -1)
    # Convert decision_function to a friendlier "anomaly score": higher = more anomalous
    anomaly_score = -raw
    threshold = float(-model.offset_) if hasattr(model, "offset_") else 0.0
    return {
        "scores": anomaly_score.tolist(),
        "flags": flags.tolist(),
        "stats": {
            "contamination": contamination,
            "n_estimators": int(model.n_estimators),
            "score_threshold": threshold,
            "flagged": int(flags.sum()),
        },
    }


def detect_iterative(values: np.ndarray, iterations: int) -> dict[str, Any]:
    """Iterative mean/std method from PDF 1.

    For i = 1..I, with f_i = (I - i) / 10:
        S_i = { d in S_{i-1} : |d - mu_{i-1}| <= f_i * sigma_{i-1} }

    A point is "removed" if it was dropped in any iteration.
    A point is "flagged for vote" only if removed while f_i >= 1.5
    (i.e. the threshold was at least 1.5 sigma).
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    iterations = int(np.clip(iterations, 1, 100))

    alive = np.ones(n, dtype=bool)
    removed_iter = np.full(n, -1, dtype=int)         # iteration when removed (1-indexed); -1 if survived
    removed_threshold = np.full(n, np.nan)           # f_i at removal time

    mu_hist: list[float] = []
    sigma_hist: list[float] = []
    f_hist: list[float] = []
    alive_count_hist: list[int] = []

    surv = values.copy()
    # Iteration 0 stats
    mu = float(np.mean(surv)) if len(surv) else 0.0
    sigma = float(np.std(surv, ddof=0)) if len(surv) else 0.0
    mu_hist.append(mu)
    sigma_hist.append(sigma)
    f_hist.append(float("nan"))  # no threshold at iter 0
    alive_count_hist.append(int(alive.sum()))

    for i in range(1, iterations + 1):
        f_i = (iterations - i) / 10.0
        if sigma == 0.0 or not np.isfinite(sigma):
            # Nothing to prune; record state and continue
            mu_hist.append(mu)
            sigma_hist.append(sigma)
            f_hist.append(f_i)
            alive_count_hist.append(int(alive.sum()))
            continue
        cutoff = f_i * sigma
        # Among currently alive points, mark those that exceed the cutoff
        alive_idx = np.where(alive)[0]
        if len(alive_idx) == 0:
            mu_hist.append(mu)
            sigma_hist.append(sigma)
            f_hist.append(f_i)
            alive_count_hist.append(0)
            continue
        diffs = np.abs(values[alive_idx] - mu)
        to_remove = alive_idx[diffs > cutoff]
        if len(to_remove):
            alive[to_remove] = False
            removed_iter[to_remove] = i
            removed_threshold[to_remove] = f_i
        # Recompute mu, sigma over survivors
        if alive.any():
            mu = float(np.mean(values[alive]))
            sigma = float(np.std(values[alive], ddof=0))
        mu_hist.append(mu)
        sigma_hist.append(sigma)
        f_hist.append(f_i)
        alive_count_hist.append(int(alive.sum()))

    total_removed_mask = ~alive
    # vote-eligible if removed while f_i >= 1.5
    vote_mask = np.zeros(n, dtype=bool)
    rt = np.nan_to_num(removed_threshold, nan=-1.0)
    vote_mask[(removed_iter > 0) & (rt >= 1.5)] = True

    # "score": the threshold (in sigmas) at which the point was killed.
    # Larger = killed earlier (more obvious outlier). Surviving points = 0.
    score = np.where(removed_iter > 0, np.nan_to_num(removed_threshold, nan=0.0), 0.0)

    # Auto-detect "go-flat" iteration K: first iter where relative sigma drop < 1%.
    K = None
    for idx in range(1, len(sigma_hist)):
        prev_s = sigma_hist[idx - 1]
        cur_s = sigma_hist[idx]
        if prev_s <= 0 or not math.isfinite(prev_s):
            continue
        rel_drop = (prev_s - cur_s) / prev_s
        if rel_drop < 0.01:
            K = idx
            break

    return {
        "scores": score.tolist(),
        "flags": vote_mask.tolist(),                # vote-eligible flags
        "removed": total_removed_mask.tolist(),     # all removals
        "stats": {
            "iterations": iterations,
            "total_removed": int(total_removed_mask.sum()),
            "flagged_for_vote": int(vote_mask.sum()),
            "final_mu": mu_hist[-1],
            "final_sigma": sigma_hist[-1],
            "initial_mu": mu_hist[0],
            "initial_sigma": sigma_hist[0],
            "K_flat": K,
        },
        "history": {
            "mu": mu_hist,
            "sigma": sigma_hist,
            "f": f_hist,
            "alive_count": alive_count_hist,
        },
    }


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def load_table(file_bytes: bytes, filename: str) -> pd.DataFrame:
    name = (filename or "").lower()
    bio = io.BytesIO(file_bytes)
    if name.endswith(".xlsx") or name.endswith(".xlsm"):
        return pd.read_excel(bio, engine="openpyxl")
    return pd.read_csv(bio)


def numeric_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def safe_float(x: Any) -> float | None:
    try:
        f = float(x)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except Exception:
        return None


def clean_for_json(obj: Any) -> Any:
    """Recursively replace NaN/Inf floats with None so strict JSON encoders accept the payload."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, (np.floating,)):
        f = float(obj)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return clean_for_json(obj.tolist())
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean_for_json(v) for v in obj]
    return obj


# --------------------------------------------------------------------------
# Analysis pipeline
# --------------------------------------------------------------------------

def run_univariate(values: np.ndarray, params: dict[str, Any]) -> dict[str, Any]:
    tau = float(params["tau"])
    k = float(params["k"])
    contamination = float(params["contamination"])
    iterations = int(params["iterations"])

    mad = detect_mad(values, tau)
    iqr = detect_iqr(values, k)
    iso = detect_isoforest(values.reshape(-1, 1), contamination)
    itr = detect_iterative(values, iterations)

    n = len(values)
    flags_matrix = np.array([
        mad["flags"], iqr["flags"], iso["flags"], itr["flags"]
    ], dtype=bool)
    vote_count = flags_matrix.sum(axis=0)
    verdict = np.where(vote_count >= 2, "high", np.where(vote_count == 1, "possible", "normal"))

    return {
        "n": int(n),
        "mode": "univariate",
        "values": values.tolist(),
        "mad": mad,
        "iqr": iqr,
        "iso": iso,
        "itr": itr,
        "vote": {
            "count": vote_count.tolist(),
            "verdict": verdict.tolist(),
            "tallies": {
                "high": int((verdict == "high").sum()),
                "possible": int((verdict == "possible").sum()),
                "normal": int((verdict == "normal").sum()),
            },
            "per_method": {
                "mad": int(flags_matrix[0].sum()),
                "iqr": int(flags_matrix[1].sum()),
                "iso": int(flags_matrix[2].sum()),
                "itr": int(flags_matrix[3].sum()),
            },
        },
    }


def run_multivariate(df: pd.DataFrame, columns: list[str], params: dict[str, Any]) -> dict[str, Any]:
    tau = float(params["tau"])
    k = float(params["k"])
    contamination = float(params["contamination"])
    iterations = int(params["iterations"])

    matrix = df[columns].to_numpy(dtype=float)
    n, p = matrix.shape

    per_col: dict[str, dict[str, Any]] = {}
    mad_row = np.zeros(n, dtype=bool)
    iqr_row = np.zeros(n, dtype=bool)
    itr_row = np.zeros(n, dtype=bool)

    # For convergence chart: normalized sigma per column
    norm_sigma_hist: dict[str, list[float]] = {}
    f_hist_shared: list[float] = []

    for col in columns:
        v = df[col].to_numpy(dtype=float)
        m = detect_mad(v, tau)
        q = detect_iqr(v, k)
        it = detect_iterative(v, iterations)
        per_col[col] = {"mad": m, "iqr": q, "itr": it}
        mad_row |= np.asarray(m["flags"], dtype=bool)
        iqr_row |= np.asarray(q["flags"], dtype=bool)
        itr_row |= np.asarray(it["flags"], dtype=bool)
        sig_hist = it["history"]["sigma"]
        s0 = sig_hist[0] if sig_hist and sig_hist[0] > 0 else 1.0
        norm_sigma_hist[col] = [(s / s0) if s0 else 0.0 for s in sig_hist]
        if not f_hist_shared:
            f_hist_shared = it["history"]["f"]

    iso = detect_isoforest(matrix, contamination)
    iso_row = np.asarray(iso["flags"], dtype=bool)

    flags_matrix = np.array([mad_row, iqr_row, iso_row, itr_row], dtype=bool)
    vote_count = flags_matrix.sum(axis=0)
    verdict = np.where(vote_count >= 2, "high", np.where(vote_count == 1, "possible", "normal"))

    # PCA to 2D for plotting
    pca_proj = None
    if p >= 2:
        # standardize columns for PCA
        col_means = matrix.mean(axis=0)
        col_stds = matrix.std(axis=0)
        col_stds = np.where(col_stds == 0, 1.0, col_stds)
        X_std = (matrix - col_means) / col_stds
        try:
            pca = PCA(n_components=2)
            pca_proj = pca.fit_transform(X_std).tolist()
        except Exception:
            pca_proj = None
    else:
        # Only one numeric column was chosen for multivariate; show value vs index
        pca_proj = [[float(i), float(matrix[i, 0])] for i in range(n)]

    return {
        "n": int(n),
        "mode": "multivariate",
        "columns": columns,
        "matrix": matrix.tolist(),
        "per_column": per_col,
        "iso": iso,
        "row_flags": {
            "mad": mad_row.tolist(),
            "iqr": iqr_row.tolist(),
            "iso": iso_row.tolist(),
            "itr": itr_row.tolist(),
        },
        "vote": {
            "count": vote_count.tolist(),
            "verdict": verdict.tolist(),
            "tallies": {
                "high": int((verdict == "high").sum()),
                "possible": int((verdict == "possible").sum()),
                "normal": int((verdict == "normal").sum()),
            },
            "per_method": {
                "mad": int(mad_row.sum()),
                "iqr": int(iqr_row.sum()),
                "iso": int(iso_row.sum()),
                "itr": int(itr_row.sum()),
            },
        },
        "pca": pca_proj,
        "convergence": {
            "f": f_hist_shared,
            "norm_sigma_per_col": norm_sigma_hist,
            "K_flat": _common_K(norm_sigma_hist),
        },
    }


def _common_K(norm_sigma_per_col: dict[str, list[float]]) -> int | None:
    """Average normalized sigma across columns, then find first iter with <1% rel drop."""
    if not norm_sigma_per_col:
        return None
    arrs = [np.asarray(v, dtype=float) for v in norm_sigma_per_col.values() if v]
    if not arrs:
        return None
    L = min(len(a) for a in arrs)
    avg = np.mean(np.stack([a[:L] for a in arrs]), axis=0)
    for idx in range(1, len(avg)):
        prev_s = avg[idx - 1]
        cur_s = avg[idx]
        if prev_s <= 0:
            continue
        rel_drop = (prev_s - cur_s) / prev_s
        if rel_drop < 0.01:
            return idx
    return None


# --------------------------------------------------------------------------
# Request models
# --------------------------------------------------------------------------

class AnalyzeRequest(BaseModel):
    session_id: str
    mode: str = Field(pattern="^(univariate|multivariate)$")
    columns: list[str]
    strictness: str = Field(default="balanced", pattern="^(lenient|balanced|strict)$")
    contamination: float = Field(default=0.05, ge=0.01, le=0.20)
    iterations: int = Field(default=20, ge=1, le=100)


class DownloadRequest(BaseModel):
    session_id: str
    action: str = Field(pattern="^(remove|flag)$")


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


def _ingest_bytes(content: bytes, filename: str) -> dict[str, Any]:
    """Parse bytes into a DataFrame, register a session, return the upload-style payload."""
    if not content:
        raise HTTPException(status_code=400, detail="Empty file.")
    try:
        df = load_table(content, filename or "")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not parse file: {exc}")
    if df.empty:
        raise HTTPException(status_code=400, detail="File parsed to zero rows.")
    cols = numeric_columns(df)
    if not cols:
        raise HTTPException(status_code=400, detail="No numeric columns detected in file.")
    sid = uuid.uuid4().hex
    SESSIONS[sid] = {
        "filename": filename,
        "df": df,
        "numeric_columns": cols,
        "result": None,
        "params": None,
    }
    preview = df.head(10).fillna("").to_dict(orient="records")
    return {
        "session_id": sid,
        "filename": filename,
        "rows": int(len(df)),
        "columns": list(df.columns),
        "numeric_columns": cols,
        "preview": preview,
    }


class LoadSampleRequest(BaseModel):
    name: str


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    content = await file.read()
    return _ingest_bytes(content, file.filename or "")


@app.post("/load-sample")
async def load_sample(req: LoadSampleRequest):
    safe = os.path.basename(req.name)
    path = os.path.join("sample_data", safe)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail=f"Unknown sample: {safe}")
    with open(path, "rb") as f:
        content = f.read()
    return _ingest_bytes(content, safe)


@app.post("/analyze")
async def analyze(req: AnalyzeRequest):
    if req.session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found. Re-upload the file.")
    session = SESSIONS[req.session_id]
    df: pd.DataFrame = session["df"]

    if not req.columns:
        raise HTTPException(status_code=400, detail="No column selected.")
    for c in req.columns:
        if c not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column {c!r} not in file.")
        if not pd.api.types.is_numeric_dtype(df[c]):
            raise HTTPException(status_code=400, detail=f"Column {c!r} is not numeric.")

    preset = STRICTNESS_PRESETS[req.strictness]
    params = {
        "tau": preset["tau"],
        "k": preset["k"],
        "contamination": float(np.clip(req.contamination, 0.01, 0.20)),
        "iterations": int(np.clip(req.iterations, 1, 100)),
        "strictness": req.strictness,
    }

    if req.mode == "univariate":
        if len(req.columns) != 1:
            raise HTTPException(status_code=400, detail="Univariate mode needs exactly one column.")
        col = req.columns[0]
        series = df[col].dropna()
        if len(series) < 3:
            raise HTTPException(status_code=400, detail="Need at least 3 non-null numeric values.")
        # Re-index df to drop NaN rows for THIS analysis
        kept_idx = series.index.to_numpy()
        values = series.to_numpy(dtype=float)
        result = run_univariate(values, params)
        result["row_index"] = kept_idx.tolist()
        result["column"] = col
    else:
        # multivariate
        sub = df[req.columns].dropna()
        if len(sub) < 3:
            raise HTTPException(status_code=400, detail="Need at least 3 complete numeric rows.")
        kept_idx = sub.index.to_numpy()
        result = run_multivariate(sub.reset_index(drop=True), req.columns, params)
        result["row_index"] = kept_idx.tolist()

    session["result"] = result
    session["params"] = params
    return clean_for_json({"session_id": req.session_id, "params": params, "result": result})


@app.post("/download/csv")
async def download_csv(req: DownloadRequest):
    if req.session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found.")
    session = SESSIONS[req.session_id]
    result = session.get("result")
    if not result:
        raise HTTPException(status_code=400, detail="No analysis result yet.")
    df: pd.DataFrame = session["df"].copy()
    kept_idx = result.get("row_index", list(range(len(df))))
    verdict = result["vote"]["verdict"]
    vote_count = result["vote"]["count"]

    # Build per-row verdict/vote columns aligned with original df via kept_idx
    df["_verdict"] = "normal"
    df["_vote_count"] = 0
    df.loc[kept_idx, "_verdict"] = verdict
    df.loc[kept_idx, "_vote_count"] = vote_count

    if req.action == "remove":
        df = df[df["_verdict"] == "normal"].drop(columns=["_verdict", "_vote_count"])
    # else flag mode keeps the columns

    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    fname = (session.get("filename") or "data") + (".cleaned.csv" if req.action == "remove" else ".flagged.csv")
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{fname}"'},
    )


@app.get("/download/report/{session_id}")
async def download_report(session_id: str):
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found.")
    session = SESSIONS[session_id]
    result = session.get("result")
    if not result:
        raise HTTPException(status_code=400, detail="No analysis result yet.")
    payload = {
        "filename": session.get("filename"),
        "params": session.get("params"),
        "result": result,
    }
    data = json.dumps(payload, indent=2, default=_json_default)
    fname = (session.get("filename") or "report") + ".outlier_report.json"
    return StreamingResponse(
        iter([data]),
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{fname}"'},
    )


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        f = float(o)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


@app.get("/sample/{name}")
async def sample(name: str):
    safe = os.path.basename(name)
    path = os.path.join("sample_data", safe)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="No such sample.")
    with open(path, "rb") as f:
        data = f.read()
    return StreamingResponse(
        iter([data]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{safe}"'},
    )


@app.get("/healthz")
async def healthz():
    return {"ok": True, "sessions": len(SESSIONS)}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
