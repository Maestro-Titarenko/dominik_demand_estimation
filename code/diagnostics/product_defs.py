"""Compare product definitions by how many products are needed to cover 95% / 90% of all
cheese units and volume (motivates the coarse 41-product definition, section 3.1).

Usage: python code/diagnostics/product_defs.py
"""
import sys, re
import numpy as np, pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]          # project root
sys.path.insert(0, str(ROOT / 'code'))
from groups import brand_group, type_group
u = pd.read_csv(ROOT / 'data' / 'upc_clean.csv')
d = u.descrip.str.lstrip('~$').str.upper()
SL = r"\bSL\b|\bSL\.|SLCD|SLICE|SLC\b|THIN|I/W|WAFER"
def form_of(seg, s):
    if seg == 'shredded_sliced':
        if re.search(r'STRING|STRG', s): return 'string'
        return 'sliced' if re.search(SL, s) else 'shredded'
    if seg == 'natural_chunk' and re.search(SL, s): return 'sliced'
    return {'natural_chunk':'chunk','grated':'grated','processed_american':'processed',
            'cream_cheese':'cream','specialty':'specialty'}[seg]
u['form'] = [form_of(g, s) for g, s in zip(u.segment, d)]
u['type'] = np.where(u.cheese_group.isin(['american/processed','cream']), u.cheese_group, u.cheese_group + ' | ' + u.form)
u['bgrp'] = u.brand.map(brand_group)
u['tgrp'] = [type_group(g, f) for g, f in zip(u.cheese_group, u.form)]
u['size_class'] = np.where(u.size_unit == 'random_weight', 'random weight',
    pd.cut(u.size_oz, [0, 6, 10, 16, 1e9], labels=['small','medium','large','bulk']).astype(str))
u['oz_eff'] = np.where(u.size_unit == 'random_weight', 12.0, u.size_oz)

w = pd.read_csv(ROOT / 'wche.csv', usecols=['UPC','MOVE','PRICE','OK'], engine='pyarrow')
w = w[(w.OK == 1) & (w.MOVE > 0) & (w.PRICE > 0)]
t = w.groupby('UPC').MOVE.sum().rename('units').reset_index().rename(columns={'UPC':'upc'})
u = u.merge(t, on='upc', how='left').fillna({'units': 0})
u['vol'] = u.units * u.oz_eff / 8                 # 8-oz equivalents (NaN for count packs)
TU, TV = u.units.sum(), u.vol.sum()
print(f"total units {TU:,.0f}; total 8oz-equiv volume {TV:,.0f} (count packs have no weight: {100*u.loc[u.size_unit=='count','units'].sum()/TU:.2f}% of units)")

ok = (u.size_unit.isin(['oz','random_weight'])) & ~u.cheese_group.isin(['not cheese'])
defs = {
  'A current: brand x type(35) x exact size (brand!=Other, type known)':
      (ok & (u.brand != 'Other') & (u.cheese_group != 'unknown'), ['brand','type','oz_eff','size_unit']),
  'B: brand x type_group(9) x size_class(5)': (ok, ['brand','tgrp','size_class']),
  'C: brand_group(8) x type_group(9) x size_class(5)': (ok, ['bgrp','tgrp','size_class']),
  'D: brand_group(8) x type_group(9) x exact size': (ok, ['bgrp','tgrp','oz_eff','size_unit']),
}
rows = []
for name, (m, keys) in defs.items():
    uu = u[m].copy()
    if 'unknown' in uu.tgrp.values: pass
    g = uu.groupby(keys, dropna=False).agg(units=('units','sum'), vol=('vol','sum'), n_upc=('upc','size')).reset_index()
    for basis, tot, col in [('units', TU, 'units'), ('volume', TV, 'vol')]:
        gs = g.sort_values(col, ascending=False)
        cum = gs[col].cumsum() / tot
        n95 = int((cum < 0.95).sum() + 1) if cum.iloc[-1] >= 0.95 else None
        rows.append({'definition': name, 'basis': basis, 'n_candidates': len(g),
                     'max_coverage_%': round(100 * cum.iloc[-1], 2),
                     'N_for_95%': n95,
                     'cov_at_N=40': round(100 * cum.iloc[min(39, len(cum)-1)], 1),
                     'cov_at_N=60': round(100 * cum.iloc[min(59, len(cum)-1)], 1),
                     'cov_at_N=80': round(100 * cum.iloc[min(79, len(cum)-1)], 1),
                     'cov_at_N=100': round(100 * cum.iloc[min(99, len(cum)-1)], 1)})
pd.set_option('display.width', 250)
print(pd.DataFrame(rows).to_string(index=False))
print('\nunits outside every definition:')
print((u[~ok].groupby(u.loc[~ok,'size_unit'] + ' / ' + u.loc[~ok,'cheese_group']).units.sum() / TU * 100).round(2).to_string())
