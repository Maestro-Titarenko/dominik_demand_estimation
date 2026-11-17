"""Reduced-form stockpiling test: post-promotion dips (section 9, item 5).

log(units) on current and lagged (1-3 weeks) promotion dummies, with store x UPC and week
fixed effects, for the 150 best-selling fixed-size UPCs; also with log price instead of the
current promotion dummy. Run on raw UPC-level movement data.

Usage: python code/diagnostics/stockpile.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]          # project root
u = pd.read_csv(ROOT / "data" / "upc_clean.csv")[["upc", "segment", "size_unit"]]
w = pd.read_csv(ROOT / "wche.csv", usecols=["STORE", "UPC", "WEEK", "MOVE", "QTY", "PRICE", "SALE", "OK"],
                engine="pyarrow")
w = w[(w.OK == 1) & (w.MOVE > 0) & (w.PRICE > 0)].merge(u, left_on="UPC", right_on="upc")
w = w[w.size_unit == "oz"]
w["p"] = w.PRICE / w.QTY
w["promo"] = (w.SALE.notna() & (w.SALE != "")).astype(float)
top = w.groupby("UPC").MOVE.sum().nlargest(150).index
w = w[w.UPC.isin(top)].sort_values(["STORE", "UPC", "WEEK"]).copy()

key = ["STORE", "UPC"]
s = w.set_index(key + ["WEEK"])
for L in [1, 2, 3]:
    lag = s[["promo"]].rename(columns={"promo": f"promo_l{L}"}).reset_index()
    lag["WEEK"] += L
    w = w.merge(lag, on=key + ["WEEK"], how="left")
w = w.dropna(subset=["promo_l1", "promo_l2", "promo_l3"])
w["ly"] = np.log(w.MOVE)
w["lp"] = np.log(w.p)


def demean(df, cols):
    x = df[cols].copy()
    g1 = df.STORE.astype(str) + "_" + df.UPC.astype(str)
    for _ in range(10):
        x = x - x.groupby(g1.values).transform("mean")
        x = x - x.groupby(df.WEEK.values).transform("mean")
    return x


def run(df, label):
    X = demean(df, ["ly", "promo", "promo_l1", "promo_l2", "promo_l3", "lp"])
    for spec in [["promo", "promo_l1", "promo_l2", "promo_l3"], ["lp", "promo_l1", "promo_l2", "promo_l3"]]:
        b, *_ = np.linalg.lstsq(X[spec].values, X["ly"].values, rcond=None)
        e = X["ly"].values - X[spec].values @ b
        n, k = X[spec].shape
        V = np.linalg.inv(X[spec].values.T @ X[spec].values) * (e @ e) / (n - k)
        print(f"{label:20s} n={n:7d} " + "  ".join(
            f"{c}={bb:+.3f}({np.sqrt(v):.3f})" for c, bb, v in zip(spec, b, np.diag(V))))


run(w, "all")
for sg, g in w.groupby("segment"):
    if len(g) > 20000:
        run(g, sg)
