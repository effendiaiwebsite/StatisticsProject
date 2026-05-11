# Algorithm Reference

Formal pseudocode for each detector, the voting ensemble, and the
multivariate per-column aggregator. Style follows the "Writing Algorithms —
Samples" PDF: numbered lines, Require / Ensure / Input / Output blocks,
one `Algorithm` per procedure.

Notation:
- `x = (x_1, ..., x_n)` is a 1-D numeric sample, `n >= 3`.
- `X` is an `n x p` numeric matrix (rows are observations, columns features).
- `1[cond]` is the indicator (1 if true, 0 if false).

---

## Algorithm 1 — Modified Z-Score (MAD)

**Input:** `x = (x_1, ..., x_n)`, threshold `tau > 0`.
**Output:** flag vector `F in {0,1}^n`, score vector `M in R^n`, stats `(median, MAD, fallback_used)`.
**Require:** `n >= 3`.
**Ensure:** if `MAD > 0`, `M_i = 0.6745 * (x_i - median) / MAD`; otherwise use the fallback below.

```
 1: med <- median(x)
 2: d_i <- |x_i - med|                                for i = 1..n
 3: MAD <- median(d_1, ..., d_n)
 4: if MAD > 0 then
 5:     M_i <- 0.6745 * (x_i - med) / MAD            for i = 1..n
 6:     fallback_used <- false
 7: else                                              // MAD = 0 fallback
 8:     mad_alt <- mean(d_1, ..., d_n)
 9:     if mad_alt = 0 then
10:         M_i <- 0                                  for i = 1..n
11:     else
12:         M_i <- 0.6745 * (x_i - med) / (1.2533 * mad_alt)
13:     fallback_used <- true
14: F_i <- 1[ |M_i| > tau ]                           for i = 1..n
15: return F, M, (med, MAD, fallback_used)
```

---

## Algorithm 2 — IQR Fences

**Input:** `x = (x_1, ..., x_n)`, multiplier `k > 0`.
**Output:** flag vector `F`, score vector `S`, stats `(Q1, Q3, IQR, lower_fence, upper_fence)`.
**Require:** `n >= 3`.
**Ensure:** `F_i = 1` iff `x_i < Q1 - k*IQR` or `x_i > Q3 + k*IQR`.

```
 1: Q1 <- 25th-percentile(x)
 2: Q3 <- 75th-percentile(x)
 3: IQR <- Q3 - Q1
 4: lo  <- Q1 - k * IQR
 5: hi  <- Q3 + k * IQR
 6: for i = 1..n do
 7:     if x_i < lo then
 8:         S_i <- (lo - x_i) / max(IQR, eps)
 9:         F_i <- 1
10:     else if x_i > hi then
11:         S_i <- (x_i - hi) / max(IQR, eps)
12:         F_i <- 1
13:     else
14:         S_i <- 0
15:         F_i <- 0
16: return F, S, (Q1, Q3, IQR, lo, hi)
```

---

## Algorithm 3 — Isolation Forest (wrapper)

**Input:** matrix `X in R^{n x p}` (use `p = 1` for the univariate case), contamination `c in [0.01, 0.20]`, random seed `r`.
**Output:** flag vector `F`, anomaly score vector `A`, stats `(c, n_estimators, score_threshold)`.
**Require:** `n >= 3`.
**Ensure:** `F_i = 1` iff the trained model predicts `x_i` as an outlier under contamination `c`.

```
 1: c <- clip(c, 0.01, 0.20)
 2: model <- IsolationForest(n_estimators = 200, contamination = c, random_state = r)
 3: model.fit(X)
 4: raw_i <- model.decision_function(X)_i              for i = 1..n   // higher = more normal
 5: A_i   <- -raw_i                                                   // higher = more anomalous
 6: pred_i <- model.predict(X)_i                       for i = 1..n   // {-1, 1}
 7: F_i <- 1[ pred_i = -1 ]                            for i = 1..n
 8: thr <- - model.offset_
 9: return F, A, (c, 200, thr)
```

---

## Algorithm 4 — Iterative Mean / Std (PDF 1)

**Input:** `x = (x_1, ..., x_n)`, iterations `I in [1, 100]`.
**Output:** vote-eligible flag vector `V in {0,1}^n`, removed mask `R in {0,1}^n`, history `(mu_0..mu_I, sigma_0..sigma_I, f_1..f_I)`, stats `(total_removed, flagged_for_vote, final_mu, final_sigma, K_flat)`.
**Require:** `n >= 3`, `I >= 1`.
**Ensure:** at iteration `i`, the survivor set `S_i` keeps only points within `f_i * sigma_{i-1}` of `mu_{i-1}` where `f_i = (I - i) / 10`. A removed point is "vote-eligible" only if `f_i >= 1.5` at removal time.

```
 1: alive_j <- 1                                       for j = 1..n
 2: rem_iter_j <- 0                                    for j = 1..n   // 0 = never removed
 3: rem_thr_j  <- NaN                                  for j = 1..n
 4: mu    <- mean(x)
 5: sigma <- std(x)                                                    // population std (ddof=0)
 6: append (mu, sigma) to history
 7: for i = 1 to I do
 8:     f_i <- (I - i) / 10
 9:     if sigma <= 0 or not finite(sigma) then
10:         append (mu, sigma, f_i) to history; continue
11:     cutoff <- f_i * sigma
12:     for each j with alive_j = 1 do
13:         if |x_j - mu| > cutoff then
14:             alive_j   <- 0
15:             rem_iter_j <- i
16:             rem_thr_j  <- f_i
17:     if any alive_j = 1 then
18:         mu    <- mean({ x_j : alive_j = 1 })
19:         sigma <- std ({ x_j : alive_j = 1 })
20:     append (mu, sigma, f_i) to history
21: R_j <- 1 - alive_j                                 for j = 1..n
22: V_j <- 1[ rem_iter_j > 0  AND  rem_thr_j >= 1.5 ]   for j = 1..n
23: K_flat <- min { i : (sigma_{i-1} - sigma_i)/sigma_{i-1} < 0.01 }  or  NIL
24: return V, R, history, (sum(R), sum(V), mu, sigma, K_flat)
```

> Note on the voting nuance. The method is designed to converge: as `i -> I`,
> `f_i -> 0`, so almost every point will eventually be "removed". Letting
> every removal vote would make this detector dominate the ensemble. We
> therefore only count a removal as a vote when the threshold `f_i` was at
> least `1.5` — i.e. the point was rejected when the boundary was still
> reasonably wide. The UI shows both `total_removed` and `flagged_for_vote`.

---

## Algorithm 5 — Voting Ensemble (Univariate)

**Input:** flag vectors `F_MAD, F_IQR, F_ISO, F_VITER in {0,1}^n` (the `V` from Algorithm 4).
**Output:** vote count `c in {0,..,4}^n`, verdict `V in {normal, possible, high}^n`.

```
 1: for i = 1..n do
 2:     c_i <- F_MAD_i + F_IQR_i + F_ISO_i + F_VITER_i
 3:     if c_i >= 2 then
 4:         V_i <- "high"
 5:     else if c_i = 1 then
 6:         V_i <- "possible"
 7:     else
 8:         V_i <- "normal"
 9: return c, V
```

---

## Algorithm 6 — Multivariate Per-Column Aggregator

For MAD, IQR, and Iterative, each column is scored independently; a row is
flagged for the method if ANY column flags it. Isolation Forest is run once
on the full numeric matrix.

**Input:** matrix `X in R^{n x p}`, column names `(c_1,..,c_p)`, params `(tau, k, c, I)`.
**Output:** row flag vectors `F_MAD, F_IQR, F_ITER`, IsoForest output `F_ISO`, vote count, verdict.

```
 1: F_MAD_i  <- 0   for i = 1..n
 2: F_IQR_i  <- 0   for i = 1..n
 3: F_ITER_i <- 0   for i = 1..n
 4: for j = 1 to p do
 5:     x^{(j)}  <- column j of X
 6:     (F_M, _, _) <- Algorithm 1 (x^{(j)}, tau)
 7:     (F_Q, _, _) <- Algorithm 2 (x^{(j)}, k)
 8:     (V_T, _, _, _) <- Algorithm 4 (x^{(j)}, I)
 9:     F_MAD  <- F_MAD  OR F_M
10:     F_IQR  <- F_IQR  OR F_Q
11:     F_ITER <- F_ITER OR V_T
12: (F_ISO, _, _) <- Algorithm 3 (X, c)                  // native multivariate
13: (count, verdict) <- Algorithm 5 (F_MAD, F_IQR, F_ISO, F_ITER)
14: return F_MAD, F_IQR, F_ISO, F_ITER, count, verdict
```

---

## Strictness presets

| Preset    | `tau` (MAD) | `k` (IQR) |
|-----------|-------------|-----------|
| Lenient   | 4.0         | 2.0       |
| Balanced  | 3.5         | 1.5       |
| Strict    | 3.0         | 1.0       |

`contamination` (Isolation Forest) and `I` (iterations) are independent
sliders. Strictness only affects MAD and IQR; contamination only affects
Isolation Forest; iterations only affects Algorithm 4.
