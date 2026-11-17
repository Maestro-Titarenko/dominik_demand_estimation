"""Random-coefficients logit demand for Dominick's cheese, estimated with pyBLP.

Model follows Nevo (2000); products, quantities and market size follow Miller and
Weinberg (2017). Full documentation: demand_estimation_nevo_style.md.

Benchmark (all defaults):
  market        store x week; store and week fixed effects absorbed
  products      41 brand group x type group x size class cells (panel_week_coarse.parquet)
  shares        8-oz equivalent units / M_s, M_s = 1.5 x max weekly store sales (all weeks)
  X1 (linear)   price ($/oz) + 19 dummies (4 size classes, 7 brand groups, 8 type groups)
  X2 (random)   constant, price, family-size package (large or bulk)
  Sigma         diagonal (constant, price, family size)
  Pi            constant x income, price x income, family size x household size
  agents        20 per store, reused every week: log income ~ store lognormal,
                household size = store mean (demographics demeaned)
  nest          none
  instruments   wholesale cost (own, rival mean in market and type group),
                BLP sums of size-class and type-group dummies (own firm / rivals),
                store demographics x size-class dummies and x number of rival products;
                screened for collinearity after absorbing FE and partialling out X1,
                then standardized
  estimator     two-step GMM, L-BFGS-B, one process

Outputs (output/): nevo_params_<tag>.csv, nevo_own_elasticities_<tag>.csv,
                   nevo_results_<tag>.pkl (full pyBLP results object)

Usage:
    python code/03_blp_nevo.py --every 1          # benchmark, full sample (~1 h, ~37 GB)
    python code/03_blp_nevo.py                    # quarterly subsample (week % 13 == 0)
    python code/03_blp_nevo.py --every 100        # smoke test, 175 markets (< 1 min)
    python code/03_blp_nevo.py --every 13 --eval-only   # one objective evaluation, timing
Robustness / alternatives:
    --mw-quantile 0.99 | --mw-mult 2 | --market-size traffic   market size
    --pi full                                                   10 demographic interactions
    --nest                                                      RC nested logit (one inside nest)
    --no-cost-iv | --no-demo-iv                                 drop instrument groups
    --panel fine --dummies fine                                 150-product panel, fine dummies
    --share-unit package                                        one package = one choice
    --fe week --store-controls                                  observed store controls, no store FE
    --promo {any,share,fw,fwdummy} [--promo-cut 0.5]            B and S promotion controls in X1
"""
import argparse
import contextlib
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyblp
from scipy.linalg import qr

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "output"
OUT.mkdir(exist_ok=True)

ap = argparse.ArgumentParser()
ap.add_argument("--every", type=int, default=13, help="keep weeks with week %% every == 0")
ap.add_argument("--fe", choices=["none", "week", "store", "store_week"], default="store_week",
                help="fixed effects absorbed into the mean utility")
ap.add_argument("--store-controls", action="store_true",
                help="add observed store characteristics to X1 (not allowed with store FE)")
ap.add_argument("--pi-constant", choices=["on", "off"], default="on",
                help="off: no demographic interactions with the constant")
ap.add_argument("--dummies", choices=["coarse", "fine"], default="coarse",
                help="coarse: 8 brand groups x 9 type groups (default); fine: 19 brands x 35 types")
ap.add_argument("--draws", type=int, default=20, help="simulated agents per market (Nevo: 20)")
ap.add_argument("--processes", type=int, default=1, help="parallel processes over markets")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--panel", choices=["coarse", "fine"], default="coarse",
                help="coarse: 41 brand-group x type-group x size-class products (90%% of units); "
                     "fine: 150 brand x type x size products")
ap.add_argument("--share-unit", choices=["package", "oz8"], default="oz8",
                help="package: one package per choice; oz8: 8-oz equivalent units "
                     "(volume-consistent with the $/oz price, as in Miller-Weinberg 2017)")
ap.add_argument("--market-size", choices=["traffic", "mw"], default="mw",
                help="traffic: weekly checkout transactions x --mkt-scale; mw: Miller-Weinberg "
                     "(2017), store-specific --mw-mult x max weekly inside sales over all weeks")
ap.add_argument("--mkt-scale", type=float, default=1.0,
                help="traffic market size = weekly checkout transactions x this factor")
ap.add_argument("--mw-mult", type=float, default=1.5, help="Miller-Weinberg multiplier (1.5)")
ap.add_argument("--mw-quantile", type=float, default=1.0,
                help="quantile of weekly store sales used in place of the max (1.0 = max)")
ap.add_argument("--pi", choices=["slim", "full"], default="slim",
                help="slim: constant x income, price x income, family_size x hhsize; "
                     "full: 10 demographic interactions")
ap.add_argument("--nest", action=argparse.BooleanOptionalAction, default=False,
                help="RCNL: all inside goods in one nest, outside good alone (drops sigma_constant)")
ap.add_argument("--demo-iv", action=argparse.BooleanOptionalAction, default=True,
                help="add store demographics x product characteristics instruments for Pi")
ap.add_argument("--cost-iv", action=argparse.BooleanOptionalAction, default=True,
                help="add wholesale-cost instruments: own cost and rivals' mean cost (market, "
                     "type group); cost = price x (1 - PROFIT/100), chain-wide per UPC-week")
ap.add_argument("--promo", choices=["none", "any", "share", "fw", "fwdummy"], default="none",
                help="promotion controls in X1, one column for bonus buy (B) and one for simple "
                     "price reduction (S): any = 1 if any UPC of the product is on the deal; "
                     "share = share of the product's units sold on the deal (moves "
                     "mechanically with quantity); fw = fixed-weight share of the product's "
                     "UPCs on the deal; fwdummy = 1[fw >= --promo-cut]")
ap.add_argument("--promo-cut", type=float, default=0.5, help="threshold for --promo fwdummy")
ap.add_argument("--eval-only", action="store_true",
                help="benchmark: evaluate the objective once at the starting values and exit")
args = ap.parse_args()
STORE_FE = args.fe in ("store", "store_week")
if args.store_controls and STORE_FE:
    ap.error("--store-controls are collinear with store fixed effects")
TAG = (f"{args.panel}_every{args.every}_fe-{args.fe}_{args.dummies}" + ("_storectrl" if args.store_controls else "")
       + ("_noPiConst" if args.pi_constant == "off" else "")
       + ("_oz8" if args.share_unit == "oz8" else "")
       + (f"_M{args.mkt_scale:g}" if (args.market_size == "traffic" and args.mkt_scale != 1.0) else "")
       + (f"_MW{args.mw_mult:g}" + (f"q{args.mw_quantile:g}" if args.mw_quantile < 1 else "")
          if args.market_size == "mw" else "")
       + f"_pi-{args.pi}"
       + ("_nest" if args.nest else "") + ("_demoiv" if args.demo_iv else "")
       + ("_costiv" if args.cost_iv else "")
       + ("" if args.promo == "none" else f"_promo-{args.promo}"
          + (f"{args.promo_cut:g}" if args.promo == "fwdummy" else "")))
WEEK_FE = args.fe in ("week", "store_week")

pyblp.options.digits = 3
pyblp.options.verbose = True


def independent_columns(a, center=True, tol=1e-8):
    """Indices of a maximal set of linearly independent columns (in original order).
    center=True checks independence after partialling out a constant."""
    if center:
        a = a - a.mean(axis=0)
    _, r, piv = qr(a, mode="economic", pivoting=True)
    rank = (np.abs(np.diag(r)) > tol * np.abs(r[0, 0])).sum()
    return np.sort(piv[:rank])


# ------------------------------------------------------------------ products
cols = ["market_ids", "store", "week", "product_ids", "brand", "firm", "type", "form",
        "cheese_group", "brand_group", "type_group", "size_class", "price", "shares",
        "units", "q_oz8", "size_oz", "random_weight", "cust", "wcost"]
if args.promo != "none":
    cols += ["promo_B", "promo_S", "promo_B_share", "promo_S_share", "promo_B_fw", "promo_S_fw"]
PANEL = DATA / ("panel_week_coarse.parquet" if args.panel == "coarse" else "panel_week.parquet")
pdat = pd.read_parquet(PANEL, columns=cols)
print(f"panel: {PANEL.name}  products: {pdat.product_ids.nunique()}")

# ------------------------------------------------------------------ demographics
demo = pd.read_stata(ROOT / "demo.dta",
                     columns=["store", "income", "incsigma", "hsizeavg", "age9", "age60"])
demo = demo.dropna().astype({"store": int})
pdat = pdat[pdat["store"].isin(demo["store"])].reset_index(drop=True)  # 10 stores lack demographics
demo = demo[demo["store"].isin(pdat["store"])].reset_index(drop=True)

# ------------------------------------------------------------------ shares
# The observed share must count choices in the same unit as the choice probability:
#   package : one choice = one package;           q = packages sold
#   oz8     : one choice = one 8-oz equivalent;   q = packages x oz / 8
# Market size (same unit as q):
#   traffic : M_st = weekly checkout transactions x mkt_scale (one occasion per trip)
#   mw      : M_s  = mw_mult x max_w Q_sw (Miller-Weinberg 2017), Q_sw = total inside
#             sales of store s in week w. Computed over ALL weeks of the panel, before
#             any subsampling, so subsample and full-sample runs share the same M.
if args.share_unit == "oz8":
    pdat["q"] = pdat["q_oz8"]          # built in 02_clean.py; random weight = 12 oz/package
else:
    pdat["q"] = pdat["units"].astype(float)
if args.market_size == "mw":
    Qsw = pdat.groupby(["store", "week"])["q"].sum()
    M = args.mw_mult * Qsw.groupby("store").quantile(args.mw_quantile)
    pdat["M"] = pdat["store"].map(M)
    m_desc = f"{args.mw_mult:g} x " + ("max" if args.mw_quantile >= 1 else
                                        f"p{100 * args.mw_quantile:g}") + " weekly store sales"
else:
    pdat["M"] = pdat["cust"] * args.mkt_scale
    m_desc = f"transactions x {args.mkt_scale:g}"
pdat["shares"] = pdat["q"] / pdat["M"]
pdat = pdat[pdat["week"] % args.every == 0]
pdat = pdat[pdat["store"].isin(demo["store"])].reset_index(drop=True)
inside = pdat.groupby("market_ids")["shares"].sum()
print(f"shares: unit={args.share_unit}, M = {m_desc}; inside share median "
      f"{inside.median():.3f}, p5 {inside.quantile(.05):.3f}, p95 {inside.quantile(.95):.3f}, "
      f"max {inside.max():.3f}")
assert inside.max() < 1, "inside share >= 1 in some market: enlarge the market size"

# ------------------------------------------------------------------ coarse groups
# brand_group (8) and type_group (9) come from the panel; definitions in code/groups.py.
# Products themselves are unchanged (150 brand x type x size cells).
BRAND_COL, TYPE_COL = (("brand_group", "type_group") if args.dummies == "coarse"
                       else ("brand", "type"))
print(f"dummies: {args.dummies}  brands: {pdat[BRAND_COL].nunique()}  "
      f"types: {pdat[TYPE_COL].nunique()}  sizes: {pdat['size_class'].nunique()}")

# ------------------------------------------------------------------ X1 dummies
# Built by hand so that collinear dummies are dropped explicitly (e.g. a brand that only
# sells one type). Size dummies form the intercept unless fixed effects are absorbed,
# in which case one size class is dropped too.
dums = pd.concat([
    pd.get_dummies(pdat["size_class"], prefix="size", prefix_sep=": ", dtype=float)
      .iloc[:, (1 if (STORE_FE or WEEK_FE) else 0):],
    pd.get_dummies(pdat[BRAND_COL], prefix="brand", prefix_sep=": ", dtype=float).iloc[:, 1:],
    pd.get_dummies(pdat[TYPE_COL], prefix="type", prefix_sep=": ", dtype=float).iloc[:, 1:],
], axis=1)
keep = independent_columns(dums.values, center=(STORE_FE or WEEK_FE))
dropped = [c for i, c in enumerate(dums.columns) if i not in set(keep)]
dums = dums.iloc[:, keep]
x1_labels = {f"x{k}": c for k, c in enumerate(dums.columns)}
dums.columns = list(x1_labels)
pdat = pd.concat([pdat, dums], axis=1)
print(f"X1 dummies: {dums.shape[1]} (dropped as collinear/base: {dropped})")

# ------------------------------------------------------------------ store controls
# Observed store characteristics, used in place of store fixed effects: price tier
# (medium = base), store size, urban location, distance to discount competitors.
if args.store_controls:
    sc = pd.read_stata(ROOT / "demo.dta", columns=["store", "priclow", "prichigh", "weekvol",
                                                    "urban", "cubdist", "omnidist"]).dropna()
    sc = sc.astype(float).astype({"store": int})
    sc["weekvol"] /= 100.0
    ctrl_labels = {"sc_priclow": "store: low price tier", "sc_prichigh": "store: high price tier",
                   "sc_weekvol": "store: weekly volume / 100", "sc_urban": "store: urban",
                   "sc_cubdist": "store: distance to Cub Foods",
                   "sc_omnidist": "store: distance to Omni"}
    sc.columns = ["store"] + list(ctrl_labels)
    n0 = pdat["store"].nunique()
    pdat = pdat.merge(sc, on="store", how="inner")
    assert pdat["store"].nunique() == n0, "store controls missing for some stores"
    x1_labels.update(ctrl_labels)
    print(f"store controls: {list(ctrl_labels.values())}")

# ------------------------------------------------------------------ promotion controls
# Treated as exogenous characteristics of the product-week (in X1, hence also in Z).
if args.promo != "none":
    src = {"any": ("promo_B", "promo_S"), "share": ("promo_B_share", "promo_S_share"),
           "fw": ("promo_B_fw", "promo_S_fw"), "fwdummy": ("promo_B_fw", "promo_S_fw")}[args.promo]
    for lab, c in zip(["B", "S"], src):
        v = pdat[c].astype(float)
        if args.promo == "fwdummy":
            v = (v >= args.promo_cut).astype(float)
        pdat[f"pr_{lab}"] = v.values
        x1_labels[f"pr_{lab}"] = f"promo {lab} ({args.promo})"
    print(f"promotion controls ({args.promo}): mean B {pdat.pr_B.mean():.3f}, "
          f"mean S {pdat.pr_S.mean():.3f}")

# ------------------------------------------------------------------ BLP instruments
# Characteristics for the instruments: size-class and type-group dummies (coarse, so the
# instrument matrix is not swamped by collinear columns).
Z = pd.concat([pd.get_dummies(pdat["size_class"], prefix="sz", dtype=float),
               pd.get_dummies(pdat["type_group"], prefix="tg", dtype=float)], axis=1)
zcols = list(Z.columns)
Z[["market_ids", "firm"]] = pdat[["market_ids", "firm"]].values
mkt_sum = Z.groupby("market_ids")[zcols].transform("sum")
firm_sum = Z.groupby(["market_ids", "firm"])[zcols].transform("sum")
iv = np.hstack([(firm_sum - Z[zcols]).values,     # firm's other products
                (mkt_sum - firm_sum).values])     # rival firms' products
n_blp = iv.shape[1]

# Demographic instruments for Pi (Miller-Weinberg 2017; Romeo 2014): store mean
# demographics (standardized) x own product characteristics (size-class dummies) and
# x the number of rival products. They vary across products within a store, so they
# survive store fixed effects.
if args.demo_iv:
    dm = demo.set_index("store")[["income", "hsizeavg", "age9", "age60"]]
    dm = (dm - dm.mean()) / dm.std()
    dstore = dm.loc[pdat["store"]].values
    chars = np.column_stack([
        pd.get_dummies(pdat["size_class"], dtype=float).iloc[:, 1:].values,
        (Z.groupby("market_ids")[zcols[0]].transform("size")                 # products in market
         - Z.groupby(["market_ids", "firm"])[zcols[0]].transform("size")).values,  # minus own firm's
    ])
    demo_iv = np.column_stack([dstore[:, [a]] * chars for a in range(dstore.shape[1])])
    iv = np.hstack([iv, demo_iv])
    print(f"demographic instruments: {demo_iv.shape[1]} candidate columns")
# Wholesale-cost instruments (supply-side cost shifters, varying by product and week):
# own wholesale cost $/oz and the mean wholesale cost of rival products in the market and
# in the same type group (the latter shift conditional shares, helping identify rho).
if args.cost_iv:
    wc = pdat["wcost"].fillna(pdat.groupby("product_ids")["wcost"].transform("median"))
    g = wc.groupby(pdat["market_ids"]); n = g.transform("size")
    rival_mkt = (g.transform("sum") - wc) / (n - 1)
    gt = wc.groupby([pdat["market_ids"], pdat["type_group"]]); nt = gt.transform("size")
    rival_type = ((gt.transform("sum") - wc) / (nt - 1).clip(lower=1)).where(nt > 1, rival_mkt)
    iv = np.hstack([np.column_stack([wc, rival_mkt, rival_type]), iv])
    print("cost instruments: own wholesale cost, rival mean cost (market, type group)")
iv = iv[:, iv.std(axis=0) > 0]
# pyBLP forms the 2SLS weighting matrix from Z = [exogenous X1 columns, excluded instruments]
# AFTER absorbing the fixed effects. Redundancy must be checked in that same space, with a
# tolerance, or near-collinear columns make (Z'Z) numerically singular (pyBLP then fails with
# "Failed to compute the 2SLS weighting matrix"; tiny numerical differences across machines
# decide which near-dependent column survives an exact-rank QR). Steps:
#   1. within-transform instruments and X1 dummies by the absorbed fixed effects;
#   2. partial the X1 dummies out of the instruments (Frisch-Waugh);
#   3. standardize and keep columns from a pivoted QR with relative tolerance IV_TOL.
IV_TOL = 1e-4


def within_fe(a):
    a = np.asarray(a, dtype=float)
    a = a - a.mean(axis=0)
    keys = (([pdat["store"].values] if STORE_FE else [])
            + ([pdat["week"].values] if WEEK_FE else []))
    for _ in range(100 if keys else 0):
        prev = a
        for k in keys:
            a = a - pd.DataFrame(a).groupby(k).transform("mean").values
        if np.abs(a - prev).max() < 1e-10:
            break
    return a


Xw = within_fe(pdat[list(x1_labels)].values)
Zw = within_fe(iv)
Zr = Zw - Xw @ np.linalg.lstsq(Xw, Zw, rcond=None)[0]
keep_var = Zr.std(axis=0) > 1e-3 * iv.std(axis=0)
_, r, piv = qr(Zr[:, keep_var] / Zr[:, keep_var].std(axis=0), mode="economic", pivoting=True)
rank = int((np.abs(np.diag(r)) > IV_TOL * np.abs(r[0, 0])).sum())
cols = np.flatnonzero(keep_var)[np.sort(piv[:rank])]
print(f"instrument screening: {iv.shape[1]} candidates -> {len(cols)} kept "
      f"({int((~keep_var).sum())} explained by FE + X1 dummies, "
      f"{int(keep_var.sum()) - len(cols)} near-collinear at tol {IV_TOL:g})")
iv = iv[:, cols]
Zfull = np.hstack([Xw, Zw[:, cols]])
Zfull = Zfull / Zfull.std(axis=0)
print(f"condition number of standardized [X1, Z] after FE: {np.linalg.cond(Zfull):.3g}")
iv = (iv - iv.mean(axis=0)) / iv.std(axis=0)      # standardize: better-conditioned GMM weights
for k in range(iv.shape[1]):
    pdat[f"demand_instruments{k}"] = iv[:, k]
print(f"BLP instruments: {iv.shape[1]} excluded instruments")

product_data = pdat.rename(columns={"price": "prices", "firm": "firm_ids"})
product_data["family_size"] = product_data["size_class"].isin(
    ["large (12-16oz)", "bulk (>16oz)"]).astype(float)
if args.nest:
    product_data["nesting_ids"] = 0          # one nest for all inside goods

# ------------------------------------------------------------------ agents
# Store-specific income distribution: lognormal with median exp(income) and standard
# deviation incsigma (dollars)  =>  sd = m * sqrt(t (t - 1)),  t = e^{s^2}.
rng = np.random.default_rng(args.seed)
R = args.draws
ratio2 = (demo["incsigma"].values / np.exp(demo["income"].values)) ** 2
demo["log_inc_sd"] = np.sqrt(np.log((1 + np.sqrt(1 + 4 * ratio2)) / 2))

agents_store = pd.concat([pd.DataFrame({
    "store": int(s["store"]),
    "income": rng.normal(s["income"], s["log_inc_sd"], R),
    "hhsize": s["hsizeavg"],
    "child": rng.binomial(1, s["age9"], R).astype(float),
    "elderly": rng.binomial(1, s["age60"], R).astype(float),
    "nodes0": rng.standard_normal(R),
    "nodes1": rng.standard_normal(R),
    "nodes2": rng.standard_normal(R),
}) for _, s in demo.iterrows()], ignore_index=True)   # same draws for a store in every week

# demean so that linear parameters describe the average consumer
for v in ["income", "hhsize", "child", "elderly"]:
    agents_store[v] -= agents_store[v].mean()
agents_store["income_sq"] = agents_store["income"] ** 2
agents_store["income_sq"] -= agents_store["income_sq"].mean()

mk = product_data[["market_ids", "store"]].drop_duplicates()
agent_data = mk.merge(agents_store, on="store").drop(columns="store")
agent_data["weights"] = 1.0 / R

print(f"markets: {len(mk):,}  stores: {mk.store.nunique()}  weeks: {product_data.week.nunique()}  "
      f"product-market obs: {len(product_data):,}  agents: {len(agent_data):,}")

# ------------------------------------------------------------------ problem
absorb = {"none": None, "week": "C(week)", "store": "C(store)",
          "store_week": "C(store) + C(week)"}[args.fe]
X1 = pyblp.Formulation("0 + prices + " + " + ".join(x1_labels), absorb=absorb)
X2 = pyblp.Formulation("1 + prices + family_size")
D = pyblp.Formulation("0 + income + income_sq + hhsize + child + elderly")
problem = pyblp.Problem((X1, X2), product_data, D, agent_data)
print(problem)

# initial values; zero entries are restrictions (pyBLP keeps them fixed at 0)
#                    income income_sq hhsize child elderly
initial_pi = np.array([
    [0.5,   0.0,   0.5,  0.5,  0.5],   # constant
    [1.0,   1.0,   1.0,  1.0,  1.0],   # price
    [0.0,   0.0,   0.5,  0.5,  0.0],   # family-size package
])
if STORE_FE:
    # hhsize is constant within a store, so constant x hhsize is a store fixed effect
    initial_pi[0, 2] = 0.0
if args.pi == "slim":
    # parsimonious Pi (store demographics are highly collinear: corr(hhsize, age9) = 0.80,
    # corr(age9, age60) = -0.71), in the spirit of Miller-Weinberg (2017), income only:
    #   constant x income, price x income, family_size x hhsize
    initial_pi = np.zeros_like(initial_pi)
    initial_pi[0, 0] = 0.5     # constant x income
    initial_pi[1, 0] = 1.0     # price x income
    initial_pi[2, 2] = 0.5     # family_size x hhsize
if args.pi_constant == "off":
    initial_pi[0, :] = 0.0
print("free pi entries (rows: constant, price, family_size):\n", (initial_pi != 0).astype(int))
initial_sigma = np.diag([0.5, 1.0, 0.5])
initial_rho = None
if args.nest:
    # rho and sigma_constant both govern inside-vs-outside substitution: keep only rho
    initial_sigma[0, 0] = 0.0
    initial_rho = 0.5
    print("nesting: one inside nest; sigma_constant fixed at 0; initial rho = 0.5")

ctx = pyblp.parallel(args.processes) if args.processes > 1 else contextlib.nullcontext()
t0 = time.time()
if args.eval_only:
    pyblp.options.verbose = False
    with ctx:
        t1 = time.time()
        problem.solve(sigma=initial_sigma, pi=initial_pi, rho=initial_rho, method="1s",
                      optimization=pyblp.Optimization("return"))
        print(f"BENCH markets={len(mk)} processes={args.processes} "
              f"one evaluation (objective + gradient + SEs): {time.time() - t1:.1f}s")
    raise SystemExit
with ctx:
    res = problem.solve(
        sigma=initial_sigma,
        pi=initial_pi,
        rho=initial_rho,
        optimization=pyblp.Optimization("l-bfgs-b", {"gtol": 1e-5}),
        method="2s",
    )
    print(res)
    print(f"estimation time: {(time.time() - t0) / 60:.1f} min")
    res.to_pickle(OUT / f"nevo_results_{TAG}.pkl")

    # ------------------------------------------------------------------ outputs
    x2_names = ["constant", "price", "family_size"]
    d_names = ["income", "income_sq", "hhsize", "child", "elderly"]
    rows = [("beta", "price", res.beta[0, 0], res.beta_se[0, 0])]
    for k, lab in enumerate(x1_labels.values(), start=1):
        rows.append(("beta", lab, res.beta[k, 0], res.beta_se[k, 0]))
    for i, n in enumerate(x2_names):
        if initial_sigma[i, i] != 0:
            rows.append(("sigma", n, res.sigma[i, i], res.sigma_se[i, i]))
    if args.nest:
        rows.append(("rho", "inside nest", res.rho[0, 0], res.rho_se[0, 0]))
    for i, n in enumerate(x2_names):
        for j, dn in enumerate(d_names):
            if initial_pi[i, j] != 0:
                rows.append(("pi", f"{n} x {dn}", res.pi[i, j], res.pi_se[i, j]))
    tab = pd.DataFrame(rows, columns=["block", "parameter", "estimate", "se"])
    tab.to_csv(OUT / f"nevo_params_{TAG}.csv", index=False)
    print(tab[tab.block != "beta"].round(3).to_string(index=False))
    print(tab[(tab.block == "beta") & (tab.parameter == "price")].round(3).to_string(index=False))

    own = res.extract_diagonals(res.compute_elasticities()).flatten()

product_data["own_elasticity"] = own
(product_data.groupby("product_ids")["own_elasticity"].median()
 .sort_values().rename("median_own_elasticity")
 .to_csv(OUT / f"nevo_own_elasticities_{TAG}.csv"))
print("\nown-price elasticity percentiles (5,25,50,75,95):",
      np.percentile(own, [5, 25, 50, 75, 95]).round(2))
print(f"share of product-markets with positive own elasticity: {(own > 0).mean():.3f}")
