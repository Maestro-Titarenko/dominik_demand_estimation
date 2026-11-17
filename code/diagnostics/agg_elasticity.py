"""Log-log price elasticities at weekly vs 4/8/13-week aggregation (stockpiling discussion).

Store x UPC and period fixed effects; aggregated price is the simple average over weeks.
150 best-selling fixed-size UPCs, raw movement data.

Usage: python code/diagnostics/agg_elasticity.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]          # project root
u = pd.read_csv(ROOT / "data" / "upc_clean.csv")[["upc", "size_unit"]]
w = pd.read_csv(ROOT / "wche.csv", usecols=["STORE", "UPC", "WEEK", "MOVE", "QTY", "PRICE", "SALE", "OK"],
                engine="pyarrow")
w = w[(w.OK == 1) & (w.PRICE > 0)].merge(u, left_on="UPC", right_on="upc")
w = w[w.size_unit == "oz"]
top = w.groupby("UPC").MOVE.sum().nlargest(150).index
w = w[w.UPC.isin(top)].copy()
w["p"] = w.PRICE / w.QTY
w["promo"] = (w.SALE.notna() & (w.SALE != "")).astype(float)


def fe_ols(df, y, xs, g1, g2):
    X = df[[y] + xs].copy()
    for _ in range(15):
        X = X - X.groupby(df[g1].values).transform("mean")
        X = X - X.groupby(df[g2].values).transform("mean")
    b, *_ = np.linalg.lstsq(X[xs].values, X[y].values, rcond=None)
    return dict(zip(xs, b.round(3)))


def run(df, label):
    df = df[df.MOVE > 0].copy()
    df["ly"], df["lp"] = np.log(df.MOVE), np.log(df.p)
    df["su"] = df.STORE.astype(str) + "_" + df.UPC.astype(str)
    print(f"{label:28s} n={len(df):8d}", fe_ols(df, "ly", ["lp"], "su", "T"),
          fe_ols(df, "ly", ["lp", "promo"], "su", "T"))


w["T"] = w.WEEK
run(w, "weekly")
for L in [4, 8, 13]:
    w["T"] = (w.WEEK - 1) // L
    a = (w.groupby(["STORE", "UPC", "T"])
          .agg(MOVE=("MOVE", "sum"), p=("p", "mean"), promo=("promo", "mean"), n=("WEEK", "size"))
          .reset_index())
    run(a[a.n >= L - 1], f"{L}-week, simple avg price")
