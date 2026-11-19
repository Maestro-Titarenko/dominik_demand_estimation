"""Section 8.5 of demand_estimation_nevo_style.md: are promotions triggered by weak demand?

Concern: cheese is perishable, so slow-selling stock may be cleared by a promotion. Then the
promotion controls would be negatively correlated with the demand shock xi.

Two checks on the coarse panel (83 stores with demographics, all weeks):

1. How much of the promotion variation is chain-wide? R2 of the fixed-weight promotion
   measures on product x week cells. Clearance of a store's own slow stock would show up as
   store-specific promotions.

2. Event study around promotion starts. A start is a week with promoted share >= 0.5 that
   follows a week with promoted share < 0.1 in the same product-store. log(8-oz quantity) is
   regressed on dummies for the 1-4 weeks before a start (weeks that are themselves not
   promoted), current promotion (B and S fixed-weight shares) and three lags of the promoted
   share, with
     (a) product-store and week fixed effects:           any pre-promotion sales slump;
     (b) product-store and product x week fixed effects: store-specific slump relative to the
         same product in other stores that week.
   Clearance predicts negative lead coefficients. Standard errors clustered by store.
   Promotions recur, so the weeks before a start are often the weeks after the previous
   promotion (post-promotion dip). "Clean" starts, with no promotion in the previous 8 weeks,
   separate the two. Product x month-of-year fixed effects remove product-specific
   seasonality; product x quarter (13-week period) fixed effects remove slow-moving
   product-specific demand.

Usage: python code/diagnostics/promo_endogeneity.py   (~3 min)
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
d = pd.read_parquet(ROOT / "data" / "panel_week_coarse.parquet",
                    columns=["store", "week", "product_ids", "q_oz8", "price",
                             "promo_B_fw", "promo_S_fw"])
demo = pd.read_stata(ROOT / "demo.dta", columns=["store", "income", "incsigma", "hsizeavg",
                                                 "age9", "age60"]).dropna()
d = d[d.store.isin(demo.store.astype(int))].reset_index(drop=True)
d["P"] = (d.promo_B_fw + d.promo_S_fw).clip(upper=1)
print(f"observations: {len(d):,}; products: {d.product_ids.nunique()}; stores: {d.store.nunique()}")

# ---------------------------------------------------------------- 1. chain-wide share
pw = d.groupby(["product_ids", "week"])
print("\n1. Share of promotion variation that is common to all stores in a product-week (R2):")
for v in ["promo_B_fw", "promo_S_fw", "P"]:
    m = pw[v].transform("mean")
    print(f"   {v:11s} {1 - ((d[v] - m) ** 2).sum() / ((d[v] - d[v].mean()) ** 2).sum():.3f}")
on = d.P >= 0.5
frac = pw["P"].transform(lambda x: (x >= 0.5).mean())
print(f"   when a product-store is promoted (share >= 0.5), the same product is promoted in "
      f"{frac[on].mean():.1%} of stores that week (median {frac[on].median():.1%})")

# ---------------------------------------------------------------- 2. event study
key = ["product_ids", "store"]
base = d[key + ["week", "P"]]
for L in [1, 2, 3]:
    lag = base.rename(columns={"P": f"P_l{L}"}).assign(week=base.week + L)
    d = d.merge(lag, on=key + ["week"], how="left")
d["start"] = ((d.P >= 0.5) & (d.P_l1 < 0.1)).astype(float)       # NaN lag -> not a start
starts = d.loc[d.start == 1, key + ["week"]]
for k in [1, 2, 3, 4]:
    lead = starts.assign(week=starts.week - k, **{f"lead{k}": 1.0})
    d = d.merge(lead, on=key + ["week"], how="left")
    d[f"lead{k}"] = d[f"lead{k}"].fillna(0.0) * (d.P < 0.1)        # only unpromoted weeks
d = d.dropna(subset=["P_l1", "P_l2", "P_l3"]).reset_index(drop=True)
d["ly"] = np.log(d.q_oz8)
d["lp"] = np.log(d.price)
print(f"\n2. Event study: {int(d.start.sum()):,} promotion starts; "
      f"{len(d):,} observations with three lags")

ps = pd.factorize(d.product_ids.astype(str) + "|" + d.store.astype(str))[0]
wk = pd.factorize(d.week)[0]
pwk = pd.factorize(d.product_ids.astype(str) + "|" + d.week.astype(str))[0]
store = pd.factorize(d.store)[0]


def within(a, groups):
    a = np.array(a, dtype=float, copy=True)
    counts = [np.bincount(g) for g in groups]
    for _ in range(500):
        prev = a.copy()
        for g, n in zip(groups, counts):
            m = np.zeros((n.size, a.shape[1]))
            np.add.at(m, g, a)
            a -= (m / n[:, None])[g]
        if np.abs(a - prev).max() < 1e-9:
            break
    return a


def ols_cluster(y, X, cl):
    b = np.linalg.solve(X.T @ X, X.T @ y)
    e = y - X @ b
    s = np.zeros((cl.max() + 1, X.shape[1]))
    np.add.at(s, cl, X * e[:, None])
    G = cl.max() + 1
    A = np.linalg.inv(X.T @ X)
    V = A @ (s.T @ s) @ A * G / (G - 1)
    return b, np.sqrt(np.diag(V))


# clean starts: promoted share < 0.1 in each of the previous 8 weeks (all 8 observed)
cnt = np.zeros(len(d))
for L in range(1, 9):
    lag = base.rename(columns={"P": "Pl"}).assign(week=base.week + L)
    cnt += (d[key + ["week"]].merge(lag, on=key + ["week"], how="left")["Pl"] < 0.1).values
d["clean"] = d.start * (cnt == 8)
cstarts = d.loc[d.clean == 1, key + ["week"]]
for k in [1, 2, 3, 4]:
    lead = cstarts.assign(week=cstarts.week - k, **{f"clead{k}": 1.0})
    d = d.merge(lead, on=key + ["week"], how="left")
    d[f"clead{k}"] = d[f"clead{k}"].fillna(0.0)
print(f"   clean starts (no promotion in the previous 8 weeks): {int(d.clean.sum()):,}")

leads = ["lead4", "lead3", "lead2", "lead1"]
cleads = ["clead4", "clead3", "clead2", "clead1"]
ctrl = ["promo_B_fw", "promo_S_fw", "P_l1", "P_l2", "P_l3"]
date = pd.Timestamp("1989-09-14") + pd.to_timedelta(7 * (d.week - 1), unit="D")
pm = pd.factorize(d.product_ids.astype(str) + "|" + date.dt.month.astype(str))[0]   # seasonality
pq = pd.factorize(d.product_ids.astype(str) + "|" + (d.week // 13).astype(str))[0]  # slow trends
specs = [
    ("(a) product-store + week FE", [ps, wk], leads, []),
    ("(a) + log price", [ps, wk], leads, ["lp"]),
    ("(a) clean starts", [ps, wk], cleads, []),
    ("(a) clean starts + product x month-of-year FE", [ps, wk, pm], cleads, []),
    ("(a) clean starts + product x quarter FE", [ps, wk, pq], cleads, []),
    ("(a) all starts + product x quarter FE", [ps, wk, pq], leads, []),
    ("(b) product-store + product x week FE", [ps, pwk], leads, []),
]
rows = {}
for label, groups, L, extra in specs:
    W = within(d[["ly"] + L + ctrl + extra].values, groups)
    b, se = ols_cluster(W[:, 0], W[:, 1:], store)
    rows[label] = [f"{bb:+.3f} ({ss:.3f})" for bb, ss in zip(b, se)][:len(L + ctrl)]
res = pd.DataFrame(rows, index=["4 weeks before", "3 weeks before", "2 weeks before",
                                "1 week before", "promo B (fw)", "promo S (fw)",
                                "promoted share lag 1", "lag 2", "lag 3"]).T
pd.set_option("display.width", 250)
print("   dependent variable: log 8-oz quantity; coefficient (s.e. clustered by store)")
print(res.to_string())
res.to_csv(ROOT / "output" / "promo_endogeneity_event_study.csv")
