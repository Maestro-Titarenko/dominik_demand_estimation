"""Feasibility check: would time-varying (state-dependent) instruments improve efficiency?

Idea under test: choose or weight instruments differently across markets/periods
(e.g. with a learned, Gumbel-Sigmoid-gated policy as in active feature acquisition).
Validity: E[xi | z, s] = 0 for exogenous market states s implies E[xi g(z, s)] = 0 for any g,
so state-dependent instruments are valid. The efficient instrument is E[dxi/dtheta | z, s]
(Chamberlain 1987). The question is how much of dxi/dtheta the instruments can predict, and
whether letting them vary with s predicts more.

Input: output/jacobian_<tag>.npz from
    python code/03_blp_nevo.py --every 4 --jacobian-at <nevo_params_*.csv>
which stores, at the estimates, the Jacobian of xi in the nonlinear parameters (sigma, pi),
prices (dxi/dalpha = -price), the X1 dummies and the screened excluded instruments Z0.

Targets (columns of the efficient-instrument problem): price and dxi/dtheta.
All targets and instruments are put in the space pyBLP estimates in: store and week fixed
effects absorbed, X1 dummies partialled out.

Instrument sets:
  A  Z0                                   (current, constant across markets)
  B  Z0, Z0 x season (3), Z0 x holiday    (vary with the time of year)
  C  B,  Z0 x year (7)                    (also vary across years)

Cross-fitting: markets are split in two folds; projection coefficients are fit on one fold
and used to predict the targets on the other, so richer sets cannot win by overfitting.
Outputs: out-of-sample R2 per target; the implied standard-error ratio relative to A, one
parameter at a time (SE ~ sqrt(P'P) / |P'T| with P the predicted target, homoskedastic; a joint
sandwich is unstable because the predicted price and sigma-price columns are almost perfectly
correlated, which is reported); relevance by season, split into total and unexplained
variation.

Usage: python code/diagnostics/iv_adaptive_feasibility.py [path/to/jacobian.npz]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
f = Path(sys.argv[1]) if len(sys.argv) > 1 else sorted((ROOT / "output").glob("jacobian_*every4_*.npz"))[0]
d = np.load(f)
names = ["price"] + [str(n) for n in d["theta_names"]]
T = np.column_stack([d["prices"], d["J"]])
Z0 = d["Z"].astype(float)
X1 = d["X1"].astype(float)
store, week = d["store"], d["week"]
N = len(T)
print(f"{f.name}: {N:,} product-markets, {len(np.unique(week))} weeks, "
      f"{Z0.shape[1]} instruments, targets: {names}")

# ---------------------------------------------------------------- market states
date = pd.Timestamp("1989-09-14") + pd.to_timedelta(7 * (week - 1), unit="D")
month = date.month.values
season = np.select([np.isin(month, [12, 1, 2]), np.isin(month, [3, 4, 5]),
                    np.isin(month, [6, 7, 8])], ["winter", "spring", "summer"], "fall")
md = date.month.values * 100 + date.day.values
holiday = ((md >= 1115) | (md <= 107)).astype(float)      # Thanksgiving to New Year
year = date.year.values
S_season = pd.get_dummies(season, dtype=float).iloc[:, 1:].values
S_year = pd.get_dummies(year, dtype=float).iloc[:, 1:].values
print(f"states: seasons {sorted(set(season))}, holiday weeks {holiday.mean():.1%} of rows, "
      f"years {year.min()}-{year.max()}")

# ---------------------------------------------------------------- within transformation
s_idx = pd.factorize(store)[0]; w_idx = pd.factorize(week)[0]
s_n = np.bincount(s_idx); w_n = np.bincount(w_idx)


def within(a):
    a = np.array(a, dtype=float, copy=True)
    a -= a.mean(axis=0)
    for _ in range(500):
        prev = a.copy()
        for idx, n in ((s_idx, s_n), (w_idx, w_n)):
            means = np.zeros((n.size, a.shape[1]))
            np.add.at(means, idx, a)
            a -= (means / n[:, None])[idx]
        if np.abs(a - prev).max() < 1e-10:
            break
    return a


Xw = within(X1)
Xq, _ = np.linalg.qr(Xw)


def resid(a):
    a = within(a)
    return a - Xq @ (Xq.T @ a)


Tr = resid(T)
inter = lambda Z, S: np.hstack([Z * S[:, [k]] for k in range(S.shape[1])])
sets = {"A": resid(Z0)}
sets["B"] = np.hstack([sets["A"], resid(inter(Z0, S_season)), resid(Z0 * holiday[:, None])])
sets["C"] = np.hstack([sets["B"], resid(inter(Z0, S_year))])
for k, Z in sets.items():
    print(f"set {k}: {Z.shape[1]} instrument columns")

# ---------------------------------------------------------------- cross-fitted projections
mkt = pd.factorize(pd.Series(store).astype(str) + "_" + pd.Series(week).astype(str))[0]
fold = np.random.default_rng(0).permutation(mkt.max() + 1)[mkt] % 2


def crossfit(Z, ridge=1e-6):
    pred = np.zeros_like(Tr)
    for k in (0, 1):
        tr, te = fold != k, fold == k
        Zt = Z[tr]
        G = Zt.T @ Zt
        G[np.diag_indices_from(G)] += ridge * np.trace(G) / G.shape[0]
        b = np.linalg.solve(G, Zt.T @ Tr[tr])
        pred[te] = Z[te] @ b
    return pred


def r2(pred, mask=slice(None)):
    e = Tr[mask] - pred[mask]
    return 1 - (e ** 2).sum(0) / (Tr[mask] ** 2).sum(0)


def se(pred):
    """One-parameter IV standard error (up to the error s.d.) with instrument pred."""
    return np.sqrt((pred ** 2).sum(0)) / np.abs((pred * Tr).sum(0))


preds = {k: crossfit(Z) for k, Z in sets.items()}
res = pd.DataFrame({f"R2 oos {k}": r2(p) for k, p in preds.items()}, index=names)
base = se(preds["A"])
for k in ("B", "C"):
    res[f"SE {k}/A"] = se(preds[k]) / base
print("\nOut-of-sample R2 of each target and implied standard-error ratio vs set A:")
print(res.round(3).to_string())

P = preds["A"]
print(f"\ncorrelation of predicted price and predicted sigma-price columns (set A): "
      f"{np.corrcoef(P[:, 0], P[:, names.index('sigma price')])[0, 1]:.3f}")

print("\nRelevance of the current instruments (set A) by state, out-of-sample R2:")
by = {}
states = [*((s, season == s) for s in ["fall", "winter", "spring", "summer"]),
          ("holiday", holiday == 1), ("non-holiday", holiday == 0)]
for lab, m in states:
    by[lab] = r2(P, m)
print(pd.DataFrame(by, index=names).round(3).to_string())
print("\nPrice after FE and dummies, by state: mean square (total) and mean squared "
      "out-of-sample residual (unexplained by set A):")
print(pd.DataFrame({lab: [m.mean(), (Tr[m, 0] ** 2).mean(), ((Tr[m, 0] - P[m, 0]) ** 2).mean()]
                    for lab, m in states}, index=["share of rows", "total", "unexplained"]).T
      .to_string(float_format=lambda v: f"{v:.5f}"))

out = ROOT / "output" / f"iv_feasibility_{f.stem.removeprefix('jacobian_')}.csv"
res.to_csv(out)
print(f"\nsaved {out.relative_to(ROOT)}")
