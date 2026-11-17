"""Promotion frequency, depth and first-stage strength of the B/S flags (section 5.3 and the
promotion discussion). Uses the fine 150-product panel.

Usage: python code/diagnostics/promo_stats.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]          # project root
pd.set_option("display.width", 200)
x = pd.read_parquet(ROOT / "data" / "panel_week.parquet")
x["bs_units"] = x.units * x.promo_BS_share

# discount: price relative to the product-store median non-promotion price
base = x[x.promo_BS == 0].groupby(["product_ids", "store"]).price.median().rename("reg")
x = x.join(base, on=["product_ids", "store"])
x["disc"] = 1 - x.price / x.reg


def tab(k):
    g = x.groupby(k)
    t = pd.DataFrame({
        "pct_obs_on_BS": 100 * g.promo_BS.mean(),
        "pct_B": 100 * g.promo_B.mean(),
        "pct_S": 100 * g.promo_S.mean(),
        "pct_units_under_BS": 100 * g.bs_units.sum() / g.units.sum(),
        "median_discount_when_BS": 100 * x[x.promo_BS == 1].groupby(k).disc.median(),
    })
    return t.sort_values("pct_obs_on_BS", ascending=False).round(1)


print("OVERALL  obs on B/S: %.1f%%  B: %.1f%%  S: %.1f%%  units under B/S: %.1f%%  median discount: %.1f%%" % (
    100 * x.promo_BS.mean(), 100 * x.promo_B.mean(), 100 * x.promo_S.mean(),
    100 * x.bs_units.sum() / x.units.sum(), 100 * x.loc[x.promo_BS == 1, "disc"].median()))
for k in ["type_group", "brand_group", "size_class"]:
    print(f"\nBY {k.upper()}")
    print(tab(k).to_string())

p = x.groupby("product_ids").promo_BS.mean()
print("\nproducts ever on B/S: %d of %d" % ((p > 0).sum(), len(p)))

# first stage: within product-store and week, log price on promo_BS
x["lp"] = np.log(x.price)
for c in ["lp", "promo_BS"]:
    x[c + "_dm"] = x[c] - x.groupby(["product_ids", "store"])[c].transform("mean")
    x[c + "_dm"] = x[c + "_dm"] - x.groupby("week")[c + "_dm"].transform("mean")
b = (x.lp_dm * x.promo_BS_dm).sum() / (x.promo_BS_dm ** 2).sum()
r2 = 1 - ((x.lp_dm - b * x.promo_BS_dm) ** 2).sum() / (x.lp_dm ** 2).sum()
print("\nwithin product-store and week, log price on promo_BS: coef %.3f, R2 %.3f" % (b, r2))
cw = x.groupby(["product_ids", "week"]).promo_BS.mean()
print("product-weeks with B/S in ALL stores %.1f%%, in NO store %.1f%%, mixed %.1f%%" % (
    100 * (cw == 1).mean(), 100 * (cw == 0).mean(), 100 * ((cw > 0) & (cw < 1)).mean()))
