"""Section 8.2 of demand_estimation_nevo_style.md.

Are price and the fixed-weight promotion measures (promo_B_fw, promo_S_fw) collinear?
Full estimation sample (83 stores with demographics), coarse panel. Within transformation as
in pyBLP: absorb store and week fixed effects, then partial out the 19 X1 dummies (FWL).
Reports correlations, R2 / VIF of price on the promotion measures, the share of price
variation left after the promotion controls, and the partial R2 of own wholesale cost for
price with and without the controls.

Usage: python code/diagnostics/promo_collinearity.py   (~2 min)
"""
import numpy as np, pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]          # project root

d = pd.read_parquet(ROOT / 'data' / 'panel_week_coarse.parquet')
demo = pd.read_stata(ROOT / 'demo.dta', columns=['store','income','incsigma','hsizeavg','age9','age60']).dropna()
d = d[d.store.isin(demo.store.astype(int))].reset_index(drop=True)
d['wcost'] = d.wcost.fillna(d.groupby('product_ids').wcost.transform('median'))   # as in 03_blp_nevo.py
dums = pd.concat([pd.get_dummies(d[c], dtype=float).iloc[:, 1:]
                  for c in ['size_class', 'brand_group', 'type_group']], axis=1).values

def within(a):
    """Absorb store and week fixed effects by alternating projections."""
    a = np.asarray(a, float); a = a - a.mean(0)
    for _ in range(200):
        prev = a
        for k in ['store', 'week']:
            a = a - pd.DataFrame(a).groupby(d[k].values).transform('mean').values
        if np.abs(a - prev).max() < 1e-9:
            break
    return a

D = within(dums)
def part(v):
    v = within(np.asarray(v, float).reshape(len(d), -1))
    return v - D @ np.linalg.lstsq(D, v, rcond=None)[0]
def r2(y, X):
    e = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    return 1 - e.var() / y.var()

p = part(d.price.values)[:, 0]
F = part(d[['promo_B_fw', 'promo_S_fw']].values)
w = part(d.wcost.values)[:, 0]
R = r2(p, F)
pr = p - F @ np.linalg.lstsq(F, p, rcond=None)[0]
wr = w - F @ np.linalg.lstsq(F, w, rcond=None)[0]
print(f"observations: {len(d):,}")
print(f"corr(price, B_fw) = {np.corrcoef(p, F[:, 0])[0, 1]:.3f}, corr(price, S_fw) = {np.corrcoef(p, F[:, 1])[0, 1]:.3f}")
print(f"R2 of price on (B_fw, S_fw) = {R:.3f};  VIF of price = {1 / (1 - R):.2f}")
print(f"share of price variation left after promotion controls = {pr.var() / p.var():.3f}")
print(f"corr(wcost, B_fw) = {np.corrcoef(w, F[:, 0])[0, 1]:.3f}, corr(wcost, S_fw) = {np.corrcoef(w, F[:, 1])[0, 1]:.3f}")
print(f"partial R2 of own wholesale cost for price: {r2(p, w[:, None]):.3f} without, "
      f"{r2(pr, wr[:, None]):.3f} with promotion controls")
