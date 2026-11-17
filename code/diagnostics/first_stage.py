"""Section 5.2 of demand_estimation_nevo_style.md.

First-stage strength of the BLP and demographic instruments for price, coarse vs fine panel.
Within transformation: partial out store FE, week FE and the X1 dummies (FWL), then regress
residualized price on residualized instruments. Reports partial R2 and F.
Quarterly subsample (week % 13 == 0). Requires both panels from 02_clean.py.

Usage: python code/diagnostics/first_stage.py
"""
import numpy as np, pandas as pd
from scipy.linalg import qr
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]          # project root
demo = pd.read_stata(ROOT / 'demo.dta', columns=['store','income','hsizeavg','age9','age60']).dropna().astype({'store':int})

def build(panel):
    x = pd.read_parquet(ROOT / 'data' / panel)
    x = x[(x.week % 13 == 0) & x.store.isin(demo.store)].reset_index(drop=True)
    Z = pd.concat([pd.get_dummies(x.size_class, prefix='sz', dtype=float),
                   pd.get_dummies(x.type_group, prefix='tg', dtype=float)], axis=1)
    zc = list(Z.columns); Z['m'] = x.market_ids; Z['f'] = x.firm
    ms = Z.groupby('m')[zc].transform('sum'); fs = Z.groupby(['m','f'])[zc].transform('sum')
    blp = np.hstack([(fs - Z[zc]).values, (ms - fs).values])
    dm = demo.set_index('store')[['income','hsizeavg','age9','age60']]; dm = (dm-dm.mean())/dm.std()
    D = dm.loc[x.store].values
    chars = np.column_stack([pd.get_dummies(x.size_class, dtype=float).iloc[:,1:].values,
                             (Z.groupby('m')[zc[0]].transform('size') - Z.groupby(['m','f'])[zc[0]].transform('size')).values])
    dmo = np.column_stack([D[:,[a]]*chars for a in range(4)])
    X1 = np.column_stack([pd.get_dummies(x.size_class, dtype=float).iloc[:,1:].values,
                          pd.get_dummies(x.brand_group, dtype=float).iloc[:,1:].values,
                          pd.get_dummies(x.type_group, dtype=float).iloc[:,1:].values])
    return x, blp, dmo, X1

def within(a, x, iters=30):
    a = a.astype(float).copy()
    for _ in range(iters):
        a -= pd.DataFrame(a).groupby(x.store.values).transform('mean').values
        a -= pd.DataFrame(a).groupby(x.week.values).transform('mean').values
    return a

def partial(y, Zx, W):
    """partial R2 and F of Zx in regression of y on [W, Zx] (all already within-transformed)."""
    b = np.linalg.lstsq(W, y, rcond=None)[0]; ry = y - W @ b
    bz = np.linalg.lstsq(W, Zx, rcond=None)[0]; rz = Zx - W @ bz
    _, r, piv = qr(rz - rz.mean(0), mode='economic', pivoting=True)
    k = (np.abs(np.diag(r)) > 1e-8*np.abs(r[0,0])).sum(); rz = rz[:, np.sort(piv[:k])]
    g = np.linalg.lstsq(rz, ry, rcond=None)[0]; e = ry - rz @ g
    r2 = 1 - (e@e)/(ry@ry); n = len(y); F = (r2/k)/((1-r2)/(n-k-W.shape[1]))
    return r2, F, k

for panel in ['panel_week.parquet', 'panel_week_coarse.parquet']:
    x, blp, dmo, X1 = build(panel)
    y = within(np.log(x.price.values)[:,None], x)[:,0]
    W = within(X1, x); B = within(blp, x); Dm = within(dmo, x)
    print(f'== {panel}: products {x.product_ids.nunique()}, markets {x.market_ids.nunique()}, '
          f'products/market sd {x.groupby("market_ids").size().std():.1f}')
    for lab, Zx in [('BLP IVs', B), ('demographic IVs', Dm), ('both', np.hstack([B, Dm]))]:
        r2, F, k = partial(y, Zx, W)
        print(f'   {lab:16s} k={k:3d}  partial R2={r2:.4f}  F={F:,.1f}')
    # how much do BLP IVs vary within product (across markets)?
    sd = pd.DataFrame(blp).groupby(x.product_ids.values).std().mean().mean()
    print(f'   mean within-product sd of BLP IVs: {sd:.3f}')

print('\n######## cost instruments (coarse panel, quarterly subsample) ########')
x, blp, dmo, X1 = build('panel_week_coarse.parquet')
x['wcost'] = x.wcost.fillna(x.groupby('product_ids').wcost.transform('median'))
g = x.groupby('market_ids').wcost; n = x.groupby('market_ids').wcost.transform('size')
x['wc_rival_mkt'] = (g.transform('sum') - x.wcost) / (n - 1)
gt = x.groupby(['market_ids', 'type_group']).wcost; nt = gt.transform('size')
x['wc_rival_type'] = ((gt.transform('sum') - x.wcost) / (nt - 1).clip(lower=1)).where(nt > 1, x.wc_rival_mkt)
y = within(np.log(x.price.values)[:, None], x)[:, 0]
yl = within(x.price.values[:, None], x)[:, 0]
W = within(X1, x)
own = within(x[['wcost']].values, x); riv = within(x[['wc_rival_mkt', 'wc_rival_type']].values, x)
B = within(blp, x); Dm = within(dmo, x)
for lab, Zx in [('own wholesale cost', own), ('rival wholesale cost (mkt, type)', riv),
                ('own + rival cost', np.hstack([own, riv])),
                ('cost + BLP', np.hstack([own, riv, B])),
                ('cost + BLP + demographic', np.hstack([own, riv, B, Dm]))]:
    r2, F, k = partial(yl, Zx, W)
    print(f'   {lab:34s} k={k:3d}  partial R2={r2:.4f}  F={F:,.1f}')
b = np.linalg.lstsq(np.hstack([W, own]), yl, rcond=None)[0]
print(f'   pass-through: d price / d wholesale cost = {b[-1]:.3f}')
