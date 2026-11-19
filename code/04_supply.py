"""Supply side: implied margins under alternative vertical conduct, and the counterfactual
that removes random-weight (store-cut) private-label cheese. Documentation: basics.md.

Demand is taken as given from a nevo_params_<tag>.csv written by 03_blp_nevo.py (same data,
market size, agents and seed). Mean utilities are recovered by share inversion, so only the
price coefficient and the nonlinear parameters are needed.

Conduct models (Villas-Boas 2007; one retailer per market, wholesale cost w observed):
  R-prod   retailer prices each product separately            p - w = -s_j / (ds_j/dp_j)
  R-brand  retailer prices brand by brand (brand managers)    ownership = brand group
  R-cat    retailer maximizes category profit                 ownership = all products
  M-bert   manufacturers set retail prices, zero retail margin (ownership = manufacturer;
           the retailer owns the private label)
  DM       double marginalization, linear wholesale prices: retailer R-cat margin plus
           manufacturer Bertrand margins in wholesale prices (through retail pass-through;
           private-label wholesale prices are not strategic)
Under R-cat with nonlinear contracts (wholesale price = manufacturer marginal cost) the total
margin is the R-cat margin: the vertically integrated / monopoly outcome.

Counterfactual: drop the private-label random-weight products. (a) prices fixed; (b) prices
re-optimized under a chosen conduct, holding fixed the marginal cost implied by that conduct
at observed prices.

Usage:
    python code/04_supply.py --params output/nevo_params_<tag>.csv [--every 1] [--processes 8]
                             [--cf-conduct R-prod] [--dm-every 13]
Outputs (output/): supply_margins_<tag>.csv, supply_cf_<tag>.csv, supply_summary_<tag>.txt
"""
import argparse
import re
import multiprocessing as mp
from pathlib import Path

import warnings

import numpy as np
import pandas as pd
from scipy.optimize import root

warnings.simplefilter('ignore')
ROOT = Path(__file__).resolve().parents[1]
DATA, OUT = ROOT / "data", ROOT / "output"
ap = argparse.ArgumentParser()
ap.add_argument("--params", required=True)
ap.add_argument("--every", type=int, default=None, help="default: the value in the params tag")
ap.add_argument("--processes", type=int, default=8)
ap.add_argument("--draws", type=int, default=20)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--mw-mult", type=float, default=1.5)
ap.add_argument("--cf-conduct", default="R-prod", choices=["R-prod", "R-brand", "R-cat", "M-bert"])
ap.add_argument("--dm-every", type=int, default=1,
                help="compute the double-marginalization margins on weeks with week %% this == 0")
args = ap.parse_args()
TAG = Path(args.params).stem.removeprefix("nevo_params_")
EVERY = args.every or int(re.search(r"every(\d+)", TAG).group(1))
PROMO = "promo-fw" in TAG

est = pd.read_csv(args.params).set_index(["block", "parameter"])["estimate"]
ALPHA = est["beta", "price"]
SIG_P, SIG_F = est["sigma", "price"], est["sigma", "family_size"]
assert est["sigma", "constant"] == 0
PI_C, PI_P, PI_F = est["pi", "constant x income"], est["pi", "price x income"], est["pi", "family_size x hhsize"]

# ------------------------------------------------------------------ data (as in 03_blp_nevo.py)
pdat = pd.read_parquet(DATA / "panel_week_coarse.parquet",
                       columns=["market_ids", "store", "week", "product_ids", "firm", "brand_group",
                                "type_group", "size_class", "random_weight", "private_label",
                                "price", "wcost", "q_oz8"])
demo = pd.read_stata(ROOT / "demo.dta", columns=["store", "income", "incsigma", "hsizeavg", "age9", "age60"])
demo = demo.dropna().astype({"store": int})
pdat = pdat[pdat.store.isin(demo.store)].reset_index(drop=True)
demo = demo[demo.store.isin(pdat.store)].reset_index(drop=True)
Qsw = pdat.groupby(["store", "week"])["q_oz8"].sum()
pdat["M"] = pdat.store.map(args.mw_mult * Qsw.groupby("store").max())
pdat["shares"] = pdat.q_oz8 / pdat.M
pdat = pdat[pdat.week % EVERY == 0].reset_index(drop=True)
pdat["wcost"] = pdat.wcost.fillna(pdat.groupby("product_ids").wcost.transform("median"))
pdat["fam"] = pdat.size_class.isin(["large (12-16oz)", "bulk (>16oz)"]).astype(float)
pdat["drop"] = ((pdat.private_label == 1) & (pdat.random_weight == 1)).astype(int)

rng = np.random.default_rng(args.seed)
R = args.draws
ratio2 = (demo.incsigma.values / np.exp(demo.income.values)) ** 2
demo["log_inc_sd"] = np.sqrt(np.log((1 + np.sqrt(1 + 4 * ratio2)) / 2))
ag = pd.concat([pd.DataFrame({
    "store": int(s["store"]), "income": rng.normal(s["income"], s["log_inc_sd"], R),
    "hhsize": s["hsizeavg"], "child": rng.binomial(1, s["age9"], R).astype(float),
    "elderly": rng.binomial(1, s["age60"], R).astype(float),
    "n0": rng.standard_normal(R), "n1": rng.standard_normal(R), "n2": rng.standard_normal(R),
}) for _, s in demo.iterrows()], ignore_index=True)
for v in ["income", "hhsize"]:
    ag[v] -= ag[v].mean()
ag["a"] = ALPHA + SIG_P * ag.n1 + PI_P * ag.income          # dU/dprice
ag["c"] = PI_C * ag.income                                   # shifts all inside goods
ag["f"] = SIG_F * ag.n2 + PI_F * ag.hhsize                   # taste for family size
AG = {s: g[["a", "c", "f"]].values for s, g in ag.groupby("store")}
print(f"{TAG}\nalpha {ALPHA:.3f}; agents with a positive price coefficient: {(ag.a > 0).mean():.1%}")

firm_code = pd.factorize(pdat.firm)[0]
brand_code = pd.factorize(pdat.brand_group)[0]
pdat["firm_c"], pdat["brand_c"] = firm_code, brand_code


# ------------------------------------------------------------------ demand primitives
def probs(delta, p, fam, A):
    """Agent choice probabilities (J x R) given mean utility delta (which includes alpha p)."""
    v = delta[:, None] + p[:, None] * (A[:, 0] - ALPHA)[None] + A[:, 1][None] + fam[:, None] * A[:, 2][None]
    m = np.maximum(v.max(0), 0)
    e = np.exp(v - m)
    return e / (np.exp(-m) + e.sum(0))


def invert(s, p, fam, A):
    d = np.log(s) - np.log(1 - s.sum())
    for _ in range(5000):
        step = np.log(s) - np.log(probs(d, p, fam, A).mean(1))
        d += step
        if np.abs(step).max() < 1e-13:
            break
    return d


def jac(S, a):
    """ds_j/dp_k (symmetric) from agent probabilities S (J x R) and price coefficients a (R)."""
    w = a / len(a)
    return np.diag(S @ w) - (S * w) @ S.T


def margin(S, a, own):
    """Bertrand-Nash margins with ownership matrix own."""
    D = jac(S, a)
    return -np.linalg.solve(own * D, S.mean(1))


def dm_manufacturer(S, a, m_r, firm, strategic):
    """Manufacturer margins under linear wholesale pricing, retailer maximizing category profit
    with retail margins m_r (Villas-Boas 2007). strategic: products with a wholesale-price setter."""
    J, n = S.shape
    D = jac(S, a)
    w2 = a ** 2 / n
    Aagg = m_r @ S                                            # R
    # H_jl = sum_k m_k d2 s_k / dp_j dp_l (category ownership)
    H = np.diag((S * (m_r[:, None] - Aagg[None])) @ w2) \
        - np.einsum("jr,lr,jlr,r->jl", S, S, m_r[:, None, None] + m_r[None, :, None] - 2 * Aagg[None, None], w2)
    G = np.linalg.solve(2 * D + H, D)                         # dp/dw: (df/dp)^-1 D
    dsdw = D @ G                                              # ds_j/dw_l
    own = (firm[:, None] == firm[None]) & strategic[:, None] & strategic[None]
    idx = np.flatnonzero(strategic)
    mw = np.zeros(J)
    mw[idx] = -np.linalg.solve((own * dsdw.T)[np.ix_(idx, idx)], S.mean(1)[idx])
    return mw, np.diag(G)


def equilibrium(delta0, p0, fam, A, c, own):
    """Bertrand-Nash prices: Morrow-Skerlos fixed point p = c + zeta(p); if it does not
    converge (inelastic demand), a quasi-Newton solve of the first-order conditions from the
    observed prices."""
    a, n = A[:, 0], len(A)
    p = p0.copy()
    for it in range(2000):
        S = probs(delta0 + ALPHA * (p - p0), p, fam, A)
        lam = S @ (a / n)
        gam = (S * (a / n)) @ S.T
        p_new = c + (own * gam) @ (p - c) / lam - S.mean(1) / lam
        if not np.all(np.isfinite(p_new)) or np.abs(p_new).max() > 100:
            break
        step = np.abs(p_new - p).max()
        p = 0.5 * p + 0.5 * p_new
        if step < 1e-10:
            return p, True

    def foc(p):
        S = probs(delta0 + ALPHA * (p - p0), p, fam, A)
        return S.mean(1) + (own * jac(S, a)) @ (p - c)

    sol = root(foc, p0, method="hybr", tol=1e-12)
    ok = bool(sol.success and np.abs(foc(sol.x)).max() < 1e-9 and (sol.x > 0).all()
              and (sol.x < 5 * p0.max()).all())
    return sol.x, ok


def surplus(delta, p, fam, A):
    """Consumer surplus in $ per unit of market size (8-oz purchase occasion) and inside share.
    Price is in $/oz and one unit is 8 oz, so the marginal utility of a dollar is -a_i / 8.
    Agents with a non-negative price coefficient are excluded from the surplus."""
    v = delta[:, None] + p[:, None] * (A[:, 0] - ALPHA)[None] + A[:, 1][None] + fam[:, None] * A[:, 2][None]
    m = np.maximum(v.max(0), 0)
    ls = m + np.log(np.exp(-m) + np.exp(v - m).sum(0))
    ok = A[:, 0] < 0
    return (8 * ls[ok] / -A[ok, 0]).sum() / len(A)


def one_market(g):
    A = AG[g.store.iloc[0]]
    a = A[:, 0]
    s, p, w, fam = g.shares.values, g.price.values, g.wcost.values, g.fam.values
    J = len(g)
    delta = invert(s, p, fam, A)
    S = probs(delta, p, fam, A)
    D = jac(S, a)
    out = pd.DataFrame(index=g.index)
    out["own_elas"] = np.diag(D) * p / s
    out["agg_elas"] = (D @ p).sum() / s.sum()        # inside volume, all prices up 1%
    owns = {"R-prod": np.eye(J), "R-brand": g.brand_c.values[:, None] == g.brand_c.values[None],
            "R-cat": np.ones((J, J)), "M-bert": g.firm_c.values[:, None] == g.firm_c.values[None]}
    for k, own in owns.items():
        out[f"m_{k}"] = margin(S, a, own)
    if g.week.iloc[0] % args.dm_every == 0:
        mw, rho = dm_manufacturer(S, a, out["m_R-cat"].values, g.firm_c.values, g.private_label.values == 0)
        out["m_DM_mfr"], out["passthrough"] = mw, rho
    # ---------------- counterfactual: remove private-label random-weight products
    keep = g["drop"].values == 0
    M = g.M.iloc[0]
    res = {"store": g.store.iloc[0], "week": g.week.iloc[0], "M": M, "n_drop": int((~keep).sum())}
    res["cs0"] = surplus(delta, p, fam, A)
    res["q0"] = s.sum()
    res["q0_drop"] = s[~keep].sum()
    res["rev0"] = 8 * p @ s
    res["pi_r0"] = 8 * (p - w) @ s
    cf = pd.DataFrame(index=g.index[keep])
    if res["n_drop"]:
        dk, pk, fk, wk = delta[keep], p[keep], fam[keep], w[keep]
        S1 = probs(dk, pk, fk, A)
        res["cs1_fix"] = surplus(dk, pk, fk, A)
        res["q1_fix"] = S1.mean(1).sum()
        res["pi_r1_fix"] = 8 * (pk - wk) @ S1.mean(1)
        cf["s_fix"] = S1.mean(1)
        own = owns[args.cf_conduct]
        c = p - out[f"m_{args.cf_conduct}"].values
        res["pi_m0"] = 8 * (p - c) @ s                       # variable profit at model costs
        res["pi_m1_fix"] = 8 * (pk - c[keep]) @ S1.mean(1)
        p1, ok = equilibrium(dk, pk, fk, A, c[keep], own[np.ix_(keep, keep)])
        res["converged"] = ok
        if ok:
            d1 = dk + ALPHA * (p1 - pk)
            S2 = probs(d1, p1, fk, A).mean(1)
            res["cs1_eq"] = surplus(d1, p1, fk, A)
            res["q1_eq"] = S2.sum()
            res["pi_r1_eq"] = 8 * (p1 - wk) @ S2
            res["pi_m1_eq"] = 8 * (p1 - c[keep]) @ S2
            cf["p_eq"], cf["s_eq"] = p1, S2
    return out, res, cf


def run_store(store):
    g = pdat[pdat.store == store]
    return [one_market(m) for _, m in g.groupby("week", sort=False)]


if __name__ == "__main__":
    with mp.get_context('fork').Pool(args.processes) as pool:
        parts = [x for chunk in pool.map(run_store, sorted(pdat.store.unique())) for x in chunk]
    mk = pd.concat([x[0] for x in parts]).loc[pdat.index]
    res = pd.DataFrame([x[1] for x in parts])
    cf = pd.concat([x[2] for x in parts])
    d = pd.concat([pdat, mk], axis=1)
    d["m_obs"] = d.price - d.wcost
    lines = []
    say = lambda *x: (print(*x), lines.append(" ".join(str(v) for v in x)))

    say(f"\nmarkets {len(res):,}; product-markets {len(d):,}")
    say("own-price elasticity percentiles (5,25,50,75,95):",
        np.percentile(d.own_elas, [5, 25, 50, 75, 95]).round(2),
        f"; share inelastic {(d.own_elas > -1).mean():.1%}")
    chk = OUT / f"nevo_own_elasticities_{TAG}.csv"
    if chk.exists():
        ref = pd.read_csv(chk, index_col=0).iloc[:, 0]
        mine = d.groupby("product_ids").own_elas.median()
        say(f"check vs pyBLP median own elasticities by product: max abs diff "
            f"{(mine - ref.reindex(mine.index)).abs().max():.2e}")

    # ---------------- margins
    models = ["R-prod", "R-brand", "R-cat", "M-bert"]
    d["m_DM"] = d["m_R-cat"] + d.get("m_DM_mfr", np.nan)
    wq = d.q_oz8
    rows = {"observed retail (p - w)": [np.median(d.m_obs / d.price), np.average(d.m_obs / d.price, weights=wq),
                                        np.nan, np.nan, np.nan, np.nan]}
    for k in models + ["DM"]:
        m = d[f"m_{k}"]
        ok = m.notna()
        lerner = (m / d.price)[ok]
        c = (d.price - m)[ok]
        e = ((m - d.m_obs) / d.price)[ok]
        rows[k] = [lerner.median(), np.average(lerner, weights=wq[ok]), (c < 0).mean(),
                   (c < d.wcost[ok]).mean(), np.sqrt((e ** 2).mean()),
                   np.corrcoef(m[ok], d.m_obs[ok])[0, 1]]
    tab = pd.DataFrame(rows, index=["median margin/p", "q-weighted margin/p", "share mc<0",
                                    "share mc<w", "RMSE vs observed (share of p)",
                                    "corr with observed margin"]).T
    say("\nImplied total margins (retail price less the marginal cost of the price-setter(s)):")
    say(tab.round(3).to_string())
    if "passthrough" in d:
        say(f"retail pass-through of own wholesale cost under R-cat (median): "
            f"{d.passthrough.median():.2f}; manufacturer margin / p under DM (median, national brands): "
            f"{(d.m_DM_mfr / d.price)[d.private_label == 0].median():.3f}")

    # Rivers-Vuong type comparison of retail conduct against observed retail margins:
    # mean difference in squared errors, standard error clustered by store.
    say("\nPairwise fit to observed retail margins (squared error difference / p^2, clustered by store):")
    for a_, b_ in [("R-prod", "R-brand"), ("R-prod", "R-cat"), ("R-brand", "R-cat"), ("M-bert", "R-prod")]:
        ea = ((d[f"m_{a_}"] - d.m_obs) / d.price) ** 2
        eb = ((d[f"m_{b_}"] - d.m_obs) / d.price) ** 2
        diff = (ea - eb)
        cl = diff.groupby(d.store).sum()
        G = len(cl)
        se = np.sqrt(G / (G - 1) * ((cl - diff.mean() * d.groupby("store").size()) ** 2).sum()) / len(d)
        say(f"  {a_:7s} vs {b_:7s}: {diff.mean():+.4f} (t = {diff.mean() / se:+.1f})  "
            f"-> {'first' if diff.mean() < 0 else 'second'} fits better")

    # by product group
    d["grp"] = np.where(d["drop"] == 1, "PL random weight", np.where(d.private_label == 1, "PL packaged", "national brands"))
    g = d.groupby("grp").apply(lambda x: pd.Series({
        "share of volume": x.q_oz8.sum() / d.q_oz8.sum(), "price $/oz": np.average(x.price, weights=x.q_oz8),
        "observed margin/p": np.average(x.m_obs / x.price, weights=x.q_oz8),
        "own elasticity (median)": x.own_elas.median(),
        **{f"{k} margin/p": np.average(x[f"m_{k}"] / x.price, weights=x.q_oz8) for k in models}}))
    say("\nBy product group (quantity-weighted):")
    say(g.round(3).T.to_string())
    d[["market_ids", "product_ids", "price", "wcost", "shares", "own_elas", "m_obs"]
      + [f"m_{k}" for k in models] + [c for c in ["m_DM_mfr", "passthrough"] if c in d]] \
        .to_csv(OUT / f"supply_margins_{TAG}.csv.gz", index=False, float_format="%.6g")

    # ---------------- counterfactual
    r = res[res.n_drop > 0].copy()
    say(f"\nCounterfactual: remove private-label random-weight products "
        f"({len(r):,} markets with such products; conduct for re-pricing: {args.cf_conduct}; "
        f"equilibrium converged in {r.converged.mean():.1%})")
    rc = r[r.converged == True]
    tot = lambda col, x=r: (x[col] * x.M).sum()
    say(f"  removed products' share of inside volume: {tot('q0_drop') / tot('q0'):.1%}")
    rowsc = {}
    for lab, suf, x in [("prices fixed", "fix", r), (f"re-priced ({args.cf_conduct})", "eq", rc)]:
        rowsc[lab] = {
            "inside volume %": 100 * (tot(f"q1_{suf}", x) / tot("q0", x) - 1),
            "diversion to remaining cheese %": 100 * (tot(f"q1_{suf}", x) - (tot("q0", x) - tot("q0_drop", x))) / tot("q0_drop", x),
            "consumer surplus %": 100 * (tot(f"cs1_{suf}", x) / tot("cs0", x) - 1),
            "consumer surplus $/store-week": ((x[f"cs1_{suf}"] - x.cs0) * x.M).mean(),
            "retail gross margin (p-w)q %": 100 * (tot(f"pi_r1_{suf}", x) / tot("pi_r0", x) - 1),
            "retail gross margin $/store-week": ((x[f"pi_r1_{suf}"] - x.pi_r0) * x.M).mean(),
            "variable profit at model costs %": 100 * (tot(f"pi_m1_{suf}", x) / tot("pi_m0", x) - 1),
        }
    say(pd.DataFrame(rowsc).round(2).to_string())
    c2 = cf.join(d[["product_ids", "price", "shares", "M", "grp", "private_label", "type_group", "q_oz8"]])
    if "p_eq" in c2:
        c2 = c2[c2.p_eq.notna()]
        c2["dp"] = c2.p_eq / c2.price - 1
        bygrp = c2.groupby("grp").apply(lambda x: pd.Series({
            "price change % (q-weighted)": 100 * np.average(x.dp, weights=x.q_oz8),
            "quantity change %, prices fixed": 100 * ((x.s_fix * x.M).sum() / (x.shares * x.M).sum() - 1),
            "quantity change %, re-priced": 100 * ((x.s_eq * x.M).sum() / (x.shares * x.M).sum() - 1)}))
        say("\n  remaining products:")
        say(bygrp.round(2).T.to_string())
        c2["same_type"] = c2.type_group.isin(["cheddar block/sliced", "other natural block/sliced",
                                              "mozzarella block/sliced", "processed"])
        byt = c2.groupby(["grp", "same_type"]).apply(lambda x: pd.Series({
            "price change %": 100 * np.average(x.dp, weights=x.q_oz8),
            "quantity change %, re-priced": 100 * ((x.s_eq * x.M).sum() / (x.shares * x.M).sum() - 1)}))
        say(byt.round(2).to_string())
    res.to_csv(OUT / f"supply_cf_{TAG}.csv", index=False)
    (OUT / f"supply_summary_{TAG}{'' if args.cf_conduct == 'R-prod' else '_' + args.cf_conduct}.txt").write_text("\n".join(lines) + "\n")
