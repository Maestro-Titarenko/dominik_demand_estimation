"""Build the store-week product panel for demand estimation.

  * market        = store x week (store-weeks with 7 days of traffic in ccount.dta)
  * product       = UPCs merged by --product-def (below); top-N products kept,
                    the rest of cheese goes to the outside good
  * price         = $ per ounce, volume-weighted over the product's UPCs
                    (random-weight items: shelf $/lb / 16)
  * quantity      = units (packages), oz, and q_oz8 = oz / 8 (8-oz equivalents;
                    random-weight packages assumed to weigh 12 oz)
  * wcost         = wholesale cost $/oz = price x (1 - PROFIT/100), volume-weighted
  * promotions    = B/S deal flags, and leave-one-out promotion rates of other products
  * shares, cust  = legacy traffic-based shares (units / weekly checkout transactions).
                    The estimation script recomputes shares from q_oz8 and the chosen
                    market size (Miller-Weinberg by default); see 03_blp_nevo.py.

Product definitions (--product-def):
  fine   : brand (19) x type (35 variety x form) x exact package size; top N = 150
  coarse : brand group (8) x type group (9) x size class (5); N chosen by --coverage
           (e.g. 0.90 = smallest N covering 90% of all cheese units)

Inputs : wche.csv, ccount.dta, data/upc_clean.csv (from 01_upc.py)
Outputs: data/products{sfx}.csv, data/markets{sfx}.csv, data/panel_week{sfx}.parquet,
         data/product_coverage{sfx}.csv   (sfx = "" for fine, "_coarse" for coarse)

Usage:
    python code/02_clean.py                                   # fine, top 150
    python code/02_clean.py --product-def coarse --coverage 0.90
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from groups import brand_group, type_group

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

ap = argparse.ArgumentParser()
ap.add_argument("--product-def", choices=["fine", "coarse"], default="fine")
ap.add_argument("--n-products", type=int, default=150, help="top-N products kept")
ap.add_argument("--coverage", type=float, default=None,
                help="keep the smallest N whose units cover this share of all cheese "
                     "units (overrides --n-products)")
args = ap.parse_args()
SFX = "" if args.product_def == "fine" else "_coarse"
RW_OZ = 12.0              # assumed oz per random-weight package ("8 TO 1" = 8 oz - 1 lb)
PRICE_OUTLIER = 3.0       # drop a UPC-store-week if unit price is >3x or <1/3 of the UPC median
CUST_OUTLIER = 2.0        # drop a store-week if traffic is >2x or <1/2 of the store median
PROMO_CODES = {"B", "C", "S"}   # documented Dominick's deal codes: bonus buy, coupon, simple reduction


def log(msg):
    print(msg, flush=True)


# =============================================================== 1. products
u = pd.read_csv(DATA / "upc_clean.csv")
d = u["descrip"].str.lstrip("~$").str.upper()

# form: refine COM_CODE segment into chunk / shredded / sliced / grated / processed / cream / specialty
SLICED = r"\bSL\b|\bSL\.|SLCD|SLICE|SLC\b|THIN|I/W|WAFER"


def form_of(seg, s):
    if seg == "shredded_sliced":
        if re.search(r"STRING|STRG", s):
            return "string"
        if re.search(SLICED, s):
            return "sliced"
        return "shredded"
    if seg == "natural_chunk" and re.search(SLICED, s):   # e.g. store-cut "DOM PP SLICED ..."
        return "sliced"
    return {"natural_chunk": "chunk", "grated": "grated", "processed_american": "processed",
            "cream_cheese": "cream", "specialty": "specialty"}[seg]


u["form"] = [form_of(g, s) for g, s in zip(u["segment"], d)]
u["type"] = np.where(
    u["cheese_group"].isin(["american/processed", "cream"]),
    u["cheese_group"],
    u["cheese_group"] + " | " + u["form"],
)

# Size is a pure product characteristic. Random-weight (store-cut, "8 TO 1" = 8oz-1lb)
# items have no fixed size: they form their own size class, flagged by a dummy.
u["random_weight"] = (u["size_unit"] == "random_weight").astype(int)
u["size_label"] = np.where(u["random_weight"] == 1, "random weight",
                           u["size_oz"].map(lambda x: f"{x:g}oz"))
u["oz_eff"] = np.where(u["random_weight"] == 1, RW_OZ, u["size_oz"])   # oz per package
u["size_class"] = np.where(
    u["random_weight"] == 1, "random weight",
    pd.cut(u["size_oz"], [0, 6, 10, 16, 1e9],
           labels=["small (<=6oz)", "medium (7-10oz)", "large (12-16oz)", "bulk (>16oz)"]).astype(str))
u["brand_group"] = u["brand"].map(brand_group)
u["type_group"] = [type_group(g, f) for g, f in zip(u["cheese_group"], u["form"])]

# eligible as (part of) an inside product. Count packs ("5 CT") have no weight and stay
# outside in both definitions.
#   fine   : identifiable brand and type, fixed oz or random weight
#   coarse : any cheese with a known weight; brand "Other" falls in "other national"
if args.product_def == "fine":
    u["eligible"] = (u["size_unit"].isin(["oz", "random_weight"]) & (u["brand"] != "Other")
                     & ~u["cheese_group"].isin(["unknown", "not cheese"]))
    u["product_id"] = np.where(
        u["eligible"], u["brand"] + " | " + u["type"] + " | " + u["size_label"], "")
else:
    u["eligible"] = (u["size_unit"].isin(["oz", "random_weight"])
                     & (u["cheese_group"] != "not cheese"))
    u["product_id"] = np.where(
        u["eligible"], u["brand_group"] + " | " + u["type_group"] + " | " + u["size_class"], "")
log(f"UPCs: {len(u)}, eligible for inside products: {u.eligible.sum()}, "
    f"distinct candidate products: {u.loc[u.eligible, 'product_id'].nunique()}")

# =============================================================== 2. weekly movement
w = pd.read_csv(
    ROOT / "wche.csv",
    usecols=["STORE", "UPC", "WEEK", "MOVE", "QTY", "PRICE", "SALE", "PROFIT", "OK"],
    engine="pyarrow",
)
log(f"\nwche raw rows: {len(w):,}")
w = w[w["OK"] == 1];                      log(f"  after OK==1:            {len(w):,}")
w = w[(w["MOVE"] > 0) & (w["PRICE"] > 0) & (w["QTY"] > 0)]
log(f"  after MOVE>0, PRICE>0:  {len(w):,}")
w = w[w["UPC"].isin(u["upc"])];           log(f"  after UPC in upc file:  {len(w):,}")

w = w.merge(u[["upc", "size_oz", "oz_eff", "size_unit", "random_weight", "eligible", "product_id"]],
            left_on="UPC", right_on="upc").drop(columns="upc")
# PRICE/QTY is the shelf price of one unit: $/package for fixed-size items and
# $/lb for random-weight items (verified: RW prices ~$2.3-3.7, /16 matches the
# $/oz of comparable Dominick's 8oz bars).
w["unit_price"] = w["PRICE"] / w["QTY"]
w["ppo"] = np.where(w["random_weight"] == 1, w["unit_price"] / 16, w["unit_price"] / w["size_oz"])
# revenue only defined for fixed-size items (RW package weights unknown)
w["rev"] = np.where(w["random_weight"] == 1, np.nan, w["unit_price"] * w["MOVE"])
# Wholesale cost per oz from Dominick's gross margin: cost = price x (1 - PROFIT/100).
# PROFIT is Dominick's average acquisition cost margin; it is identical across stores in a
# UPC-week (chain-wide cost). Margins outside (0, 100) are treated as missing.
w["wc_ok"] = (w["PROFIT"] > 0) & (w["PROFIT"] < 100)
w["wc_ppo"] = np.where(w["wc_ok"], w["ppo"] * (1 - w["PROFIT"] / 100), np.nan)
w["promo"] = w["SALE"].isin(PROMO_CODES)
w["promo_B"] = (w["SALE"] == "B").astype(int)     # bonus buy
w["promo_S"] = (w["SALE"] == "S").astype(int)     # simple price reduction

# price outliers relative to the UPC's own median shelf price
med = w.groupby("UPC")["unit_price"].transform("median")
bad = (w["unit_price"] > PRICE_OUTLIER * med) | (w["unit_price"] < med / PRICE_OUTLIER)
log(f"  price outliers dropped: {bad.sum():,} rows ({100*w.loc[bad,'MOVE'].sum()/w.MOVE.sum():.3f}% of units)")
w = w[~bad]

# =============================================================== 3. market size from ccount
c = pd.read_stata(ROOT / "ccount.dta", columns=["store", "week", "custcoun"])
log(f"\nccount raw rows (store-days): {len(c):,}")
c = c[c["week"].between(1, 399) & c["store"].between(1, 500) & (c["custcoun"] > 0)]
log(f"  after valid week/store/custcoun>0: {len(c):,}")
m = (c.groupby(["store", "week"])
       .agg(cust=("custcoun", "sum"), ndays=("custcoun", "size"))
       .reset_index())
m["week"] = m["week"].astype(int)
m["store"] = m["store"].astype(int)
log(f"  store-weeks: {len(m):,}; with 7 days: {(m.ndays == 7).sum():,}")
m = m[m["ndays"] == 7]
store_med = m.groupby("store")["cust"].transform("median")
bad_m = (m["cust"] > CUST_OUTLIER * store_med) | (m["cust"] < store_med / CUST_OUTLIER)
log(f"  traffic outliers dropped: {bad_m.sum():,}")
m = m[~bad_m].drop(columns="ndays")

# keep markets present in both sources
cheese_tot = (w.groupby(["STORE", "WEEK"])
                .agg(units_all=("MOVE", "sum"), rev_all=("rev", "sum"))
                .reset_index()
                .rename(columns={"STORE": "store", "WEEK": "week"}))
mk = m.merge(cheese_tot, on=["store", "week"], how="inner")
log(f"\nmarkets (store-weeks) in both wche and ccount: {len(mk):,} "
    f"({mk.store.nunique()} stores, weeks {mk.week.min()}-{mk.week.max()})")
mk["all_cheese_share"] = mk["units_all"] / mk["cust"]
log(f"  all-cheese share of traffic: median {mk.all_cheese_share.median():.3f}, "
    f"max {mk.all_cheese_share.max():.3f}")
w = w.merge(mk[["store", "week"]], left_on=["STORE", "WEEK"], right_on=["store", "week"]).drop(
    columns=["store", "week"])

# =============================================================== 4. aggregate UPCs -> products
we = w[w["eligible"]].copy()
we["oz"] = we["MOVE"] * we["oz_eff"]
we["ppo_x_oz"] = we["ppo"] * we["oz"]
we["wc_x_oz"] = np.where(we["wc_ok"], we["wc_ppo"] * we["oz"], 0.0)
we["oz_wc"] = np.where(we["wc_ok"], we["oz"], 0.0)
we["promo_units"] = we["MOVE"] * we["promo"]
we["promo_B_units"] = we["MOVE"] * we["promo_B"]
we["promo_S_units"] = we["MOVE"] * we["promo_S"]
# Fixed-weight promotion intensity: each UPC is weighted by its share of the product's
# units in that store over the whole sample (time-invariant), so the measure does not
# move mechanically with this week's quantities (unlike the unit-weighted share).
#   promo_B_fw = sum_u w_us 1[u on B] / sum_u w_us, over the product's UPCs sold that week
we["w_fw"] = (we.groupby(["product_id", "STORE", "UPC"])["MOVE"].transform("sum")
              / we.groupby(["product_id", "STORE"])["MOVE"].transform("sum"))
we["fw_B"] = we["w_fw"] * we["promo_B"]
we["fw_S"] = we["w_fw"] * we["promo_S"]
pm = (we.groupby(["product_id", "STORE", "WEEK"])
        .agg(units=("MOVE", "sum"), oz=("oz", "sum"), ppo_x_oz=("ppo_x_oz", "sum"),
             wc_x_oz=("wc_x_oz", "sum"), oz_wc=("oz_wc", "sum"),
             promo_units=("promo_units", "sum"),
             promo_B_units=("promo_B_units", "sum"), promo_S_units=("promo_S_units", "sum"),
             promo_B=("promo_B", "max"), promo_S=("promo_S", "max"),
             fw_B=("fw_B", "sum"), fw_S=("fw_S", "sum"), w_fw=("w_fw", "sum"),
             n_upc_sold=("UPC", "nunique"))
        .reset_index())

# rank candidate products by total units over all markets
rank = (pm.groupby("product_id")
          .agg(units=("units", "sum"), n_markets=("units", "size"))
          .sort_values("units", ascending=False)
          .reset_index())
rank["rank"] = np.arange(1, len(rank) + 1)
total_units = w["MOVE"].sum()
total_traffic = mk["cust"].sum()
rank["cum_units_share_of_cheese"] = rank["units"].cumsum() / total_units

cov = []
for n in sorted({10, 20, 30, 40, 50, 60, 80, 100, 150, 200, 250, 300, len(rank)} & set(range(1, len(rank) + 1))):
    top = rank.head(n)
    cov.append({
        "N": n,
        "pct_of_all_cheese_units": 100 * top.units.sum() / total_units,
        "inside_share_mean": top.units.sum() / total_traffic,
        "min_markets_of_Nth_product": int(top.n_markets.iloc[-1]),
    })
cov = pd.DataFrame(cov)
cov.to_csv(DATA / f"product_coverage{SFX}.csv", index=False)
log("\ncoverage by number of inside products (share of ALL cheese incl. random-weight):")
log(cov.round(3).to_string(index=False))

if args.coverage is not None:
    N_PRODUCTS = int((rank["cum_units_share_of_cheese"] < args.coverage).sum() + 1)
else:
    N_PRODUCTS = args.n_products
keep = set(rank.head(N_PRODUCTS)["product_id"])
pm = pm[pm["product_id"].isin(keep)].copy()
log(f"\nproduct definition: {args.product_def}; kept N = {N_PRODUCTS} products covering "
    f"{100 * rank['cum_units_share_of_cheese'].iloc[N_PRODUCTS - 1]:.1f}% of all cheese units")

# =============================================================== 5. panel
pm = pm.rename(columns={"STORE": "store", "WEEK": "week"}).merge(
    mk[["store", "week", "cust"]], on=["store", "week"])
pm["market_ids"] = pm["store"].astype(str) + "_" + pm["week"].astype(str)
# $ per ounce: volume-weighted mean over merged UPCs (= revenue / ounces). For the fine
# definition all UPCs in a product share one size, so this equals the unit-weighted mean.
pm["price"] = pm["ppo_x_oz"] / pm["oz"]
pm["q_oz8"] = pm["oz"] / 8.0          # quantity in 8-oz equivalent units
# wholesale cost $/oz, volume-weighted over UPCs with a valid margin (NaN if none)
pm["wcost"] = np.where(pm["oz_wc"] > 0, pm["wc_x_oz"] / pm["oz_wc"].clip(lower=1e-12), np.nan)
pm["promo_share"] = pm["promo_units"] / pm["units"]
# B/S promotion flags at the store-week-product level. A product can merge several UPCs:
#   promo_B, promo_S : 1 if any of its UPCs carried that deal code in the store-week
#   promo_BS         : 1 if any UPC was on B or S
#   promo_BS_share   : share of the product's units sold under a B or S code
pm["promo_BS"] = ((pm["promo_B"] + pm["promo_S"]) > 0).astype(int)
pm["promo_BS_share"] = (pm["promo_B_units"] + pm["promo_S_units"]) / pm["units"]
#   promo_B_share, promo_S_share : share of the product's units sold under B / under S
#   promo_B_fw, promo_S_fw       : fixed-weight share of the product's UPCs on B / on S
pm["promo_B_share"] = pm["promo_B_units"] / pm["units"]
pm["promo_S_share"] = pm["promo_S_units"] / pm["units"]
pm["promo_B_fw"] = pm["fw_B"] / pm["w_fw"]
pm["promo_S_fw"] = pm["fw_S"] / pm["w_fw"]
pm["shares"] = pm["units"] / pm["cust"]

prod = u[u["product_id"].isin(keep)].merge(
    we.groupby("UPC")["MOVE"].sum().rename("upc_units"), left_on="upc", right_index=True,
    how="left").fillna({"upc_units": 0})


def modal(col):
    """Unit-weighted most common value of `col` within each product."""
    return (prod.groupby(["product_id", col])["upc_units"].sum().reset_index()
                .sort_values("upc_units").drop_duplicates("product_id", keep="last")
                .set_index("product_id")[col])


if args.product_def == "coarse":
    attrs = (prod.groupby("product_id")
                 .agg(n_upc=("upc", "size"),
                      upcs=("upc", lambda x: " ".join(map(str, x))),
                      descrips=("descrip", lambda x: " / ".join(sorted(set(x)))),
                      brand_group=("brand_group", "first"), type_group=("type_group", "first"),
                      size_class=("size_class", "first"))
                 .reset_index()
                 .merge(rank[["product_id", "rank", "units", "n_markets"]], on="product_id")
                 .sort_values("rank"))
    for col in ["firm", "cheese_group", "form"]:            # modal value within the cell
        attrs[col] = attrs["product_id"].map(modal(col))
    attrs["brand"] = attrs["brand_group"]
    attrs["type"] = attrs["type_group"]
    attrs["random_weight"] = (attrs["size_class"] == "random weight").astype(int)
    vol = prod.assign(v=prod["upc_units"] * prod["size_oz"]).groupby("product_id")
    attrs["size_oz"] = attrs["product_id"].map(vol["v"].sum() / vol["upc_units"].sum())
    attrs.loc[attrs["random_weight"] == 1, "size_oz"] = np.nan   # volume-weighted mean size
    attrs["private_label"] = (attrs["brand_group"] == "private label").astype(int)
    attrs.to_csv(DATA / f"products{SFX}.csv", index=False)
else:
  attrs = (prod.groupby("product_id")
               .agg(brand=("brand", "first"), firm=("firm", "first"), type=("type", "first"),
                    cheese_group=("cheese_group", "first"), form=("form", "first"),
                    size_oz=("size_oz", "first"), random_weight=("random_weight", "first"),
                    size_label=("size_label", "first"), n_upc=("upc", "size"),
                    upcs=("upc", lambda x: " ".join(map(str, x))),
                    descrips=("descrip", lambda x: " / ".join(sorted(set(x)))))
               .reset_index()
               .merge(rank[["product_id", "rank", "units", "n_markets"]], on="product_id")
               .sort_values("rank"))
  attrs["private_label"] = (attrs["firm"] == "Dominick's (PL)").astype(int)
  attrs["brand_group"] = attrs["brand"].map(brand_group)
  attrs["type_group"] = [type_group(g, f) for g, f in zip(attrs["cheese_group"], attrs["form"])]
  # size class (categorical): random weight is its own class
  attrs["size_class"] = np.where(
      attrs["random_weight"] == 1, "random weight",
      pd.cut(attrs["size_oz"], [0, 6, 10, 16, 1e9],
             labels=["small (<=6oz)", "medium (7-10oz)", "large (12-16oz)", "bulk (>16oz)"]).astype(str))
  attrs.to_csv(DATA / f"products{SFX}.csv", index=False)

pm = pm.merge(attrs[["product_id", "brand", "firm", "type", "cheese_group", "form",
                     "size_oz", "random_weight", "size_class", "private_label",
                     "brand_group", "type_group"]], on="product_id")
pm = pm.rename(columns={"product_id": "product_ids"})
cols = ["market_ids", "store", "week", "product_ids", "brand", "firm", "type",
        "cheese_group", "form", "size_oz", "random_weight", "size_class", "private_label",
        "brand_group", "type_group",
        "price", "wcost", "units", "oz", "q_oz8", "shares", "promo_share",
        "promo_B", "promo_S", "promo_BS", "promo_BS_share", "promo_B_share", "promo_S_share",
        "promo_B_fw", "promo_S_fw", "n_upc_sold", "cust"]
pm = pm[cols].sort_values(["store", "week", "product_ids"]).reset_index(drop=True)

inside = pm.groupby("market_ids")["shares"].sum().rename("inside_share")
mk["market_ids"] = mk["store"].astype(str) + "_" + mk["week"].astype(str)
mk = mk.merge(inside, on="market_ids", how="left").fillna({"inside_share": 0})
mk["outside_share"] = 1 - mk["inside_share"]
mk["n_products"] = mk["market_ids"].map(pm.groupby("market_ids").size()).fillna(0).astype(int)
# drop incomplete store-weeks: fewer than half the store's usual number of products
thin = mk["n_products"] < 0.5 * mk.groupby("store")["n_products"].transform("median")
log(f"\nincomplete markets dropped (n_products < 50% of store median): {thin.sum()}")
mk = mk[~thin & (mk["n_products"] > 0)]
pm = pm[pm["market_ids"].isin(mk["market_ids"])].copy()
assert (mk["inside_share"] < 1).all(), "inside share >= 1 in some market"

# Promotion environment of each product in its store-week: share of the OTHER products
# (leave-one-out) on a B/S deal, overall and within the same type group, brand group and
# size class. n_oth_* counts those other products; the rate is 0 when there are none.
for name, keys in [("mkt", ["market_ids"]),
                   ("type", ["market_ids", "type_group"]),
                   ("brand", ["market_ids", "brand_group"]),
                   ("size", ["market_ids", "size_class"])]:
    g = pm.groupby(keys)["promo_BS"]
    n_oth = g.transform("size") - 1
    k_oth = g.transform("sum") - pm["promo_BS"]
    pm[f"n_oth_{name}"] = n_oth
    pm[f"promo_rate_oth_{name}"] = np.where(n_oth > 0, k_oth / n_oth.clip(lower=1), 0.0)

pm.to_parquet(DATA / f"panel_week{SFX}.parquet", index=False)
mk[["market_ids", "store", "week", "cust", "units_all", "rev_all", "all_cheese_share",
    "inside_share", "outside_share", "n_products"]].to_csv(DATA / f"markets{SFX}.csv", index=False)

# =============================================================== 6. summary
log("\n================ summary ================")
log(f"panel rows (product-market): {len(pm):,}")
log(f"markets: {len(mk):,}   stores: {mk.store.nunique()}   products: {pm.product_ids.nunique()}")
log(f"products per market: mean {mk.n_products.mean():.1f}, min {mk.n_products.min()}, max {mk.n_products.max()}")
log(f"inside share:  mean {mk.inside_share.mean():.3f}, p5 {mk.inside_share.quantile(.05):.3f}, "
    f"p95 {mk.inside_share.quantile(.95):.3f}, max {mk.inside_share.max():.3f}")
log(f"outside share: mean {mk.outside_share.mean():.3f}")
log(f"product share: median {pm.shares.median():.5f}, min {pm.shares.min():.2e}")
log(f"price ($/oz): median {pm.price.median():.3f}, p1 {pm.price.quantile(.01):.3f}, p99 {pm.price.quantile(.99):.3f}")
log("\nunits of cheese in kept markets that go to the outside option, by reason:")
wk = w.merge(mk[["store", "week"]], left_on=["STORE", "WEEK"], right_on=["store", "week"])
wk["reason"] = np.select(
    [wk["size_unit"].eq("count"),
     ~wk["eligible"],
     ~wk["product_id"].isin(keep)],
    ["count pack (no weight)", "not eligible (brand Other / unknown type / not cheese)",
     "outside top N"],
    "inside product")
log((wk.groupby("reason")["MOVE"].sum() / wk["MOVE"].sum() * 100).round(2).to_string())
log("\ninside products by size class:")
log(attrs.groupby("size_class").agg(n=("product_id", "size"), units=("units", "sum"))
         .assign(pct_units=lambda x: (100 * x.units / x.units.sum()).round(1)).to_string())
