# Demand Estimation, Nevo Style: Random-Coefficients Logit on Dominick's Cheese

This note estimates a random-coefficients logit (BLP) demand system for packaged cheese at
Dominick's Finer Foods. The model follows Nevo (2000), *A Practitioner's Guide to Estimation
of Random-Coefficients Logit Models of Demand*, JEMS 9(4). Product definition, quantity units
and market size follow Miller and Weinberg (2017), *Understanding the Price Effects of the
MillerCoors Joint Venture*, Econometrica 85(6).

- Data construction: [code/01_upc.py](code/01_upc.py) (UPC attributes),
  [code/02_clean.py](code/02_clean.py) (store-week panel), [code/groups.py](code/groups.py)
  (brand and type groups)
- Estimation: [code/03_blp_nevo.py](code/03_blp_nevo.py), pyBLP 1.3.0
- Diagnostics behind the numbers quoted below: [code/diagnostics/](code/diagnostics/)
- Cluster scripts: [code/longleaf/](code/longleaf/)
- Environment: [requirements.txt](requirements.txt) (Python 3.12, used for the benchmark),
  [requirements-py39.txt](requirements-py39.txt) (Python 3.9); setup steps in [README.md](README.md)

---

## 1. Benchmark specification at a glance

| Component | Benchmark choice |
|---|---|
| Market | store × week: 27,230 markets, 83 stores with demographics, 380 weeks between September 1989 and April 1997 |
| Product | brand group (8) × type group (9) × size class (5); top 41 products, covering 90.5% of all cheese units |
| Quantity | 8-oz equivalent units (total ounces ÷ 8) |
| Price | dollars per ounce, volume-weighted over the UPCs in a product |
| Market size | $M_s = 1.5 \times \max_w Q_{sw}$: 1.5 times the store's largest weekly inside sales, in 8-oz units (Miller–Weinberg) |
| Outside good | no purchase, plus cheese outside the 41 products |
| Model | random-coefficients logit, **no nest** |
| Fixed effects | store and week, absorbed |
| Linear characteristics $X_1$ | price + 19 dummies (size class, brand group, type group) |
| Random coefficients $X_2$ | constant, price, family-size package |
| Demographic interactions $\Pi$ | constant × income, price × income, family size × household size |
| Instruments | 51 candidates: wholesale cost (3), BLP sums (28), demographics × characteristics (20); screened to a well-conditioned set (49 in the full sample) |
| Estimator | two-step GMM, nested fixed point (Nevo 2000 appendix) |

Command: `python code/03_blp_nevo.py --every 1` (all other options are the defaults).

---

## 2. Mapping to Nevo (2000) and Miller–Weinberg (2017)

| | Nevo (2000): cereal | Miller–Weinberg (2017): beer | This note: cheese |
|---|---|---|---|
| Market | city × quarter | region × month | store × week |
| Products | 24 brands | ~39 brand × package size | 41 brand group × type group × size class |
| Quantity | servings | 144-oz equivalents | 8-oz equivalents |
| Market size | population × 1 serving/day | 1.5 × max regional sales | 1.5 × max weekly store sales |
| Model | RC logit | RC nested logit | RC logit (nest not identified here, §6) |
| Fixed effects | brand | product + time | store + week |
| Demographics | income, income², age, child | income | income, household size |
| Price instruments | Hausman | distance × diesel, post-merger dummy | wholesale cost |
| Π instruments | — | mean income × characteristics | store demographics × characteristics |

---

## 3. Data

### 3.1 Products

A product is a cell of three coarse characteristics. Every UPC with a known weight is mapped
to one cell; UPCs in the same cell are merged.

**Brand groups (8)**

| Brand group | Brands |
|---|---|
| private label | Dominick's, Heritage House, Generic |
| Kraft | Kraft, Cracker Barrel, Kraft Healthy Favorites, Light n' Lively |
| Philadelphia | Philadelphia |
| Velveeta | Velveeta |
| Sargento | Sargento, MooTown |
| County Line | County Line |
| Borden | Borden |
| other national | all other brands, incl. Healthy Choice, Land O Lakes, Laughing Cow and brands coded "Other" |

**Type groups (9):** cream cheese; processed; shredded; cheddar block/sliced; mozzarella
block/sliced; other natural block/sliced; grated parmesan; snack (string, crackers); specialty.

**Size classes (5):** small (≤6 oz); medium (7–10 oz); large (12–16 oz); bulk (>16 oz);
random weight (store-cut packages labelled "8 TO 1", i.e. 8 oz–1 lb).

**Selection.** Cells are ranked by total units over the sample, and the smallest set covering
90% of all cheese units is kept: **41 products covering 90.5%**. Count packs ("5 CT", no
weight; 0.8% of units) and non-cheese items (0.2%) are never inside goods. The full list, with
member UPCs and descriptions, is in `data/products_coarse.csv`.

**Why coarse products.** The cardinality of every characteristic is at most 10, as advised.
Compared with the earlier 150 brand × type × exact-size products (93.7% coverage), the panel
shrinks from 3.28 million to 1.12 million product-market rows, and estimation is about three
times faster.

### 3.2 Quantities, prices and shares

- **Quantity** $q_{jt}$: total ounces sold ÷ 8. Random-weight packages are assumed to weigh
  12 oz, the midpoint of 8 oz–1 lb.
- **Price** $p_{jt}$: total revenue ÷ total ounces, in $/oz. For random-weight items the shelf
  price is per pound, so $/oz = shelf price ÷ 16.
- **Share** $s_{jt} = q_{jt} / M_s$.

The share unit (8-oz equivalents) matches the price unit ($/oz), as in Miller–Weinberg. Each
choice is "buy one 8-oz unit of cheese, or not". Package size enters only as a product
characteristic.

### 3.3 Market size

$$M_s = 1.5 \times \max_w Q_{sw}, \qquad Q_{sw} = \sum_{j} q_{jsw}.$$

$M_s$ is computed over all weeks of the full panel before any subsampling, so subsample and
full-sample runs share the same market size. Across stores, $M_s$ ranges from 6,373 to 24,998
8-oz units per week (mean 14,282). Inside shares in the full sample: mean 0.283, median 0.271,
5th percentile 0.168, 95th percentile 0.431, maximum 0.667 (= 1/1.5 by construction, reached
in each store's peak week).

**Caveat.** In 62 of 93 stores the maximum week is week 223 (December 16, 1993), a chain-wide
pre-Christmas promotion: chain sales were 2.6 times a normal week, and Philadelphia 8-oz cream
cheese sold 15 times its usual volume. `--mw-quantile 0.99` replaces the maximum by the 99th
percentile as a robustness check.

An alternative market size, weekly checkout transactions from `ccount.dta`, is available with
`--market-size traffic`.

### 3.4 Other variables in the panel

| Variable | Definition |
|---|---|
| `wcost` | wholesale cost $/oz $= p \times (1 - \text{PROFIT}/100)$, volume-weighted; margins outside (0, 100) treated as missing (0.05% of rows) |
| `promo_B`, `promo_S`, `promo_BS` | 1 if any UPC in the product carried a bonus-buy (B) or simple price reduction (S) code |
| `promo_BS_share` | share of the product's units sold under B or S |
| `promo_B_share`, `promo_S_share` | share of the product's units sold under B, under S |
| `promo_B_fw`, `promo_S_fw` | fixed-weight share of the product's UPCs on B, on S (§8.2) |
| `promo_rate_oth_*` | share of the *other* products on B/S in the same market, type group, brand group or size class |

### 3.5 Descriptive statistics (estimation sample)

**Sample.** 27,230 store-weeks (83 stores × 380 weeks, September 14, 1989 to April 30, 1997);
41 products; 1,087,156 product-market observations; 39.9 products per market on average
(minimum 28).

**Product-market variables**

| Variable | Mean | SD | 10th pct. | Median | 90th pct. |
|---|---|---|---|---|---|
| price ($/oz) | 0.245 | 0.098 | 0.162 | 0.223 | 0.373 |
| wholesale cost ($/oz) | 0.161 | 0.070 | 0.101 | 0.147 | 0.238 |
| retail margin, $1 - \text{wcost}/p$ | 0.340 | 0.101 | 0.229 | 0.339 | 0.464 |
| on B/S promotion (any UPC) | 0.318 | 0.466 | 0 | 0 | 1 |
| share of units sold on B/S | 0.190 | 0.331 | 0 | 0 | 0.840 |
| product share $s_{jt}$ | 0.0071 | | | 0.0034 | |
| inside share $\sum_j s_{jt}$ (per market) | 0.283 | | 0.168 | 0.271 | |

Promotion frequency is higher here than at the UPC level (19%) because a product is flagged
when *any* of its UPCs is on promotion.

**Store demographics (83 stores).** Median household income $41,578 (log income: mean 10.62,
SD 0.28 across stores); average household size: mean 2.66, SD 0.26.

**Volume shares, median price and promotion frequency by characteristic**

| Brand group | Products | Volume share | Median $/oz | Share of weeks on B/S |
|---|---|---|---|---|
| private label | 13 | 39.1% | 0.195 | 0.27 |
| Kraft | 12 | 27.5% | 0.257 | 0.34 |
| Philadelphia | 3 | 15.3% | 0.173 | 0.28 |
| Sargento | 4 | 6.0% | 0.304 | 0.29 |
| Velveeta | 2 | 5.9% | 0.177 | 0.43 |
| other national | 4 | 3.5% | 0.289 | 0.40 |
| County Line | 2 | 1.5% | 0.266 | 0.33 |
| Borden | 1 | 1.2% | 0.202 | 0.47 |

| Type group | Products | Volume share | Median $/oz | Share of weeks on B/S |
|---|---|---|---|---|
| processed | 9 | 34.2% | 0.192 | 0.38 |
| cream cheese | 5 | 21.0% | 0.170 | 0.32 |
| shredded | 9 | 17.1% | 0.257 | 0.28 |
| cheddar block/sliced | 4 | 10.2% | 0.231 | 0.34 |
| other natural block/sliced | 4 | 9.7% | 0.235 | 0.34 |
| mozzarella block/sliced | 3 | 3.8% | 0.198 | 0.19 |
| grated parmesan | 3 | 2.8% | 0.408 | 0.32 |
| snack (string, crackers) | 3 | 0.8% | 0.354 | 0.24 |
| specialty | 1 | 0.4% | 0.422 | 0.52 |

| Size class | Products | Volume share | Median $/oz | Share of weeks on B/S |
|---|---|---|---|---|
| medium (7–10 oz) | 17 | 41.0% | 0.251 | 0.41 |
| large (12–16 oz) | 10 | 38.5% | 0.197 | 0.42 |
| random weight | 4 | 14.0% | 0.193 | 0.00 |
| small (≤6 oz) | 9 | 3.5% | 0.332 | 0.17 |
| bulk (>16 oz) | 1 | 3.0% | 0.162 | 0.32 |

Volume shares are of the 41 inside products' total 8-oz volume. Price per ounce falls with
package size (small $0.33, medium $0.25, large $0.20, bulk $0.16), and random-weight store-cut
cheese is almost never on B/S promotion.

---

## 4. Model

### 4.1 Utility

$$
u_{ijt} = \underbrace{\alpha p_{jt} + x_j\beta + \lambda_s + \tau_w + \xi_{jt}}_{\delta_{jt}}
+ \underbrace{[\,1,\ p_{jt},\ \text{family}_j\,]\,(\Pi D_i + \Sigma\nu_i)}_{\mu_{ijt}}
+ \varepsilon_{ijt},
$$

with the outside good normalized to $u_{i0t} = \varepsilon_{i0t}$ and $\varepsilon$ i.i.d. type-I
extreme value.

- $\delta_{jt}$ is the mean utility: price, the 19 characteristic dummies $x_j$, store fixed
  effects $\lambda_s$, week fixed effects $\tau_w$ and the structural error $\xi_{jt}$.
- $\mu_{ijt}$ is the consumer-specific deviation. $[1, p_{jt}, \text{family}_j]$ are the three
  characteristics with random coefficients: the constant (taste for buying cheese at all),
  price (price sensitivity) and the family-size dummy (large or bulk package).

### 4.2 Linear part $X_1$

Price, plus 19 dummies: 4 size classes, 7 brand groups and 8 type groups, each relative to a
base category. Store and week fixed effects are absorbed.

### 4.3 Random coefficients

$\Sigma$ is diagonal with three entries (constant, price, family size).

$\Pi$ keeps three interactions:

| $X_2$ \ $D$ | income | household size |
|---|---|---|
| constant | ✓ | 0 † |
| price | ✓ | — |
| family size | — | ✓ |

† Household size is constant within a store, so constant × household size would be a store
fixed effect.

Each entry has a direct interpretation: price × income measures how price sensitivity changes
with income, the core interaction in both Nevo (2000) and Miller–Weinberg (2017); constant ×
income measures how the propensity to buy cheese changes with income; family size × household
size measures whether larger households prefer larger packages.

**Why only three.** Store demographics are highly collinear across the 83 stores:
corr(household size, share of children) = 0.80 and corr(share of children, share of elderly)
= −0.71. With ten interactions, the estimates were large, offset each other in sign, and were
all insignificant (§6). Miller–Weinberg use income only.

`--pi full` restores ten interactions: constant × (income, child, elderly), price × (income,
income², household size, child, elderly), family size × (household size, child).

### 4.4 Fixed effects and random coefficients

Store fixed effects and random coefficients coexist. Fixed effects shift the *level* of mean
utility; random coefficients govern *substitution*, through price and product characteristics,
which vary across products within a store. Only an interaction of the constant with a
demographic that is constant within a store (here household size) is absorbed by store fixed
effects. Nevo (2000) likewise combines brand fixed effects with random coefficients.

### 4.5 Simulated consumers

Each store has 20 simulated consumers, reused in every week:

| Variable | Construction |
|---|---|
| `income` | log income $\sim N(\texttt{income}_s, s_s^2)$; $\texttt{income}_s$ is the store's log median household income, and $s_s$ matches the dollar standard deviation `incsigma` under a lognormal ($s_s \approx 0.5$) |
| `hhsize` | store average household size, the same for every consumer in the store |
| `nodes0`–`nodes2` | $\nu_i \sim N(0, I_3)$ |

Demographics are demeaned over all agents, so $(\alpha, \beta)$ are the tastes of the average
consumer. (`income_sq`, `child`, `elderly` are also drawn; they are used only with `--pi full`.)

---

## 5. Instruments

51 candidate excluded instruments:

| Group | Count | Construction | Mainly identifies |
|---|---|---|---|
| Wholesale cost | 3 | own wholesale cost $/oz; mean wholesale cost of rival products in the market; mean wholesale cost of rival products in the same type group | price |
| BLP sums | 28 | sums of size-class (5) and type-group (9) dummies over the firm's other products and over rival firms' products | random coefficients |
| Demographic | 20 | store mean income, household size, share of children, share of elderly (standardized) × 4 size-class dummies and × the number of rival products | $\Pi$ |

**Screening.** pyBLP builds the 2SLS weighting matrix from $Z = [X_1, \text{excluded instruments}]$
after absorbing the fixed effects, so redundancy is checked in that space: instruments are
within-transformed by store and week, the 19 dummies are partialled out, and columns are kept
from a pivoted QR with relative tolerance $10^{-4}$, then standardized. The number kept depends
on the sample: 49 in the full sample, 46 on the quarterly subsample (2,170 markets), 35 on the
175-market smoke test. The condition number of the standardized $[X_1, Z]$ is 763 in the full
sample.

### 5.1 Wholesale cost

Dominick's records each UPC's gross margin `PROFIT`, so wholesale cost is
$p_{jt}(1 - \text{PROFIT}_{jt}/100)$. It is a supply-side cost shifter that varies by product
and week, the analogue of Miller–Weinberg's distance × diesel price:

- within a UPC-store, its time-series variation is as large as that of the retail price
  (standard deviation of logs 0.099 vs 0.100), with a correlation of only 0.36;
- it is identical across stores in a given UPC-week (a chain-wide cost), so it survives week
  fixed effects through cross-product variation;
- a 1-cent increase in wholesale cost raises the retail price by 1.27 cents on average.

**Validity concerns.** `PROFIT` is based on Dominick's average acquisition cost, not the
current replacement cost. Manufacturer trade deals can lower the wholesale cost and trigger a
retail promotion at the same time, and promotions also raise demand directly through displays
and feature advertising. Both would correlate the instrument with $\xi_{jt}$. It is still the
best cost shifter available in these data.

### 5.2 First-stage strength

Partial $R^2$ and $F$ of the excluded instruments for price, after partialling out store and
week fixed effects and the 19 dummies (quarterly subsample, 2,170 markets):

| Instruments | 150-product panel | 41-product panel |
|---|---|---|
| BLP sums only | $R^2$ = 0.037, F = 377 | $R^2$ = 0.006, F = 23 |
| demographic only | $R^2$ = 0.0005, F = 6.6 | $R^2$ = 0.001, F = 4.6 |
| own wholesale cost | — | $R^2$ = **0.835**, F = 438,441 |
| all 46 | — | $R^2$ = 0.839, F = 9,775 |

**Why BLP instruments alone fail on the 41-product panel.** With time-invariant
characteristics and store, week and characteristic fixed effects, BLP sums vary only when the
choice set changes. The 41 products sell in almost every store-week (standard deviation of the
number of products per market: 0.7, vs 4.4 with 150 products), so the BLP sums are nearly
constant within a product and carry almost no information about price. A time-varying cost
shifter is required, which is what Miller–Weinberg use.

### 5.3 Instruments not used

- **Hausman** (prices of the same product in other stores): all stores are in Chicago and
  prices are set chain-wide, so other-store prices share the demand shocks.
- **Own promotion flags (B/S):** conditional on price, promotion weeks sell about 26% more,
  so promotions shift demand directly and fail the exclusion restriction.

---

## 6. Specification history and diagnostics

All smoke tests use weeks 100, 200 and 300: 175 markets, 83 stores.

| Run | Products | Shares, market size | Nest | Instruments | Price coef. | Median own elasticity | Positive elasticities | Notes |
|---|---|---|---|---|---|---|---|---|
| 1 | 150 | packages, transactions | no | BLP | −11.3 (7.2) | −5.32 | 10.9% | 10 Π, large offsetting signs |
| 2 | 41 | 8-oz, MW | yes | BLP + demographic | +0.50 (1.2) | +11.0 | 100% | ρ = 0.99 (bound); negative step-1 objective |
| 3 | 41 | 8-oz, MW | yes | + wholesale cost | −0.06 (0.6) | −2.17 | 17.3% | ρ = 0.99 (bound) |
| 4 | 41 | 8-oz, MW | no | + wholesale cost | −12.9 (2.4) | −2.83 | 0% | 10 Π, all insignificant |
| 5 | 41 | 8-oz, MW | no | + wholesale cost | −19.4 (4.0) | −3.40 | 1.0% | same as 4, different collinear column kept; unstable |
| 6 | 41 | 8-oz, MW | no | + wholesale cost | −10.7 (5.5) | −2.40 | 0% | 3 Π |
| **7** | **41** | **8-oz, MW** | **no** | **+ wholesale cost, screened** | **−10.7 (5.0)** | **−2.38** | **0%** | **benchmark: 3 Π; weighting-matrix condition number 2×10⁶** |

**Lessons.**

1. **Wholesale cost is essential** once products are coarse (runs 2 → 3).
2. **The nest is not identified here** (runs 3 vs 4). The nesting parameter needs variation
   that moves total inside share separately from shares among inside goods, e.g. changes in
   the number of products. The choice set is nearly fixed and $M_s$ is fixed per store, so ρ
   runs to its upper bound of 0.99. Miller–Weinberg also report a weak first stage for their
   nesting parameter.
3. **Ten Π interactions are not identified** from 83 stores with collinear demographics
   (runs 4–5); three interactions are (run 6).

**Numerical issue, resolved in run 7.** In runs 1–6 the condition number of the GMM weighting
matrix was of order 10^18–10^19. The instruments had been screened for collinearity *before*
absorbing the fixed effects and without partialling out the dummies, so near-dependent columns
survived. On Longleaf (newer numpy/scipy) one more such column survived, and pyBLP failed with
"Failed to compute the 2SLS weighting matrix". Screening in the absorbed, dummy-partialled space
(§5) brings the condition number to about 2×10⁶, with essentially unchanged estimates.

---

## 7. Benchmark results

Both runs use the defaults in §1, on Longleaf, one core each.

| | Quarterly subsample | Full sample |
|---|---|---|
| Markets / stores / weeks | 2,170 / 83 / 30 | 27,230 / 83 / 380 |
| Product-market observations | 86,595 | 1,087,156 |
| Excluded instruments after screening | 46 | 49 |
| Run time / peak memory | 5.5 min / 3.1 GB | 64 min / 37 GB |

Both GMM steps converge. In the full sample, step 2 stops after 18 iterations, when the
objective improves by less than $10^{-5}$ (relative), with a projected gradient norm of 0.11 on
an objective of $1.6\times10^4$. The reduced Hessian has one zero eigenvalue, from
$\sigma_{\text{constant}}$ sitting at its lower bound of 0. The weighting-matrix condition
number is $2.6\times10^6$.

### 7.1 Parameter estimates

Standard errors in parentheses. Price is in dollars per ounce.

| Parameter | Smoke test (175 markets) | Quarterly (2,170) | **Full (27,230)** |
|---|---|---|---|
| price (mean $\alpha$) | −10.66 (4.97) | −7.46 (1.52) | **−8.21 (0.45)** |
| $\sigma$ constant | 0.00 (34.1) | 0.00 (0.74) | **0.00 (0.21)** |
| $\sigma$ price | 0.00 (9.39) | 2.65 (0.87) | **2.93 (0.24)** |
| $\sigma$ family size | 0.00 (1.65) | 0.32 (0.34) | **0.33 (0.10)** |
| $\pi$ constant × income | 0.21 (53.6) | 0.56 (0.83) | **0.31 (0.25)** |
| $\pi$ price × income | −0.46 (3.82) | −0.20 (0.27) | **−0.12 (0.08)** |
| $\pi$ family size × household size | 0.33 (0.10) | 0.38 (0.03) | **0.36 (0.01)** |

Linear characteristics, full sample (base categories: bulk size, Borden, cheddar block/sliced):

| Size class | | Brand group | | Type group | |
|---|---|---|---|---|---|
| small | −1.17 (0.03) | Sargento | 1.68 (0.01) | specialty | 1.74 (0.03) |
| medium | 0.19 (0.03) | Philadelphia | 1.59 (0.01) | grated parmesan | 0.35 (0.03) |
| large | 0.34 (0.01) | private label | 1.19 (0.02) | processed | −0.04 (0.01) |
| random weight | 0.26 (0.02) | Kraft | 0.99 (0.01) | other natural block/sliced | −0.05 (0.00) |
| | | Velveeta | 0.87 (0.01) | snack | −0.18 (0.02) |
| | | other national | 0.06 (0.01) | shredded | −0.33 (0.01) |
| | | County Line | −0.08 (0.01) | cream cheese | −0.79 (0.02) |
| | | | | mozzarella block/sliced | −1.22 (0.00) |

### 7.2 Own-price elasticities (full sample)

Across all product-markets: 5th percentile −2.41, 25th −1.80, **median −1.50**, 75th −1.28,
95th −1.00. No product-market has a positive own-price elasticity.

| Most elastic (median) | | Least elastic (median) | |
|---|---|---|---|
| Kraft grated parmesan, small | −2.41 | private label cream cheese, medium | −0.92 |
| other national specialty, small | −2.28 | Philadelphia cream cheese, medium | −1.10 |
| Kraft grated parmesan, medium | −2.22 | private label processed, large | −1.14 |
| Sargento snack, small | −2.14 | Velveeta processed, bulk | −1.16 |
| other national snack, small | −2.03 | Philadelphia cream cheese, large | −1.25 |
| Kraft snack, small | −1.97 | private label processed, random weight | −1.28 |

Full product-level results: `output/nevo_own_elasticities_coarse_every1_*.csv`.

### 7.3 Interpretation

1. **Stable across samples.** The quarterly and full-sample estimates are close; the full
   sample mainly shrinks the standard errors (price: 1.52 → 0.45).
2. **Price.** The mean price coefficient is −8.2. With $\sigma_{\text{price}} = 2.93$, the price
   coefficients of the simulated consumers have mean −8.2 and standard deviation 2.9; only
   0.26% of consumers have a positive price coefficient, so the model no longer produces
   positive own elasticities.
3. **Elasticities are moderate**, median −1.5, below Miller–Weinberg's beer estimates
   (−4.3 to −6.1). They are highest for small, high-price-per-ounce products (grated parmesan,
   snacks) and lowest for the large staples (cream cheese, processed slices). Because quantity
   is measured in 8-oz units and the products are coarse aggregates, these are elasticities of
   cell-level volume, which are naturally lower than UPC-level elasticities.
4. **Heterogeneity.** Random coefficients on price and on family size are significant.
   Family size × household size is strongly positive (0.36, t ≈ 45): stores with larger
   households buy relatively more large packages. Constant × income is not significant.
   Price × income is negative and only marginally significant (t ≈ −1.6), i.e. if anything
   higher-income stores are slightly *more* price sensitive, the opposite of the usual sign.
   Store demographics vary little across 83 stores, which limits what Π can pick up.
5. **$\sigma_{\text{constant}} = 0$.** No unobserved heterogeneity in the taste for cheese
   relative to the outside good beyond what store fixed effects and income capture.
6. **Linear characteristics.** Among brand groups, Sargento, Philadelphia and private label
   have the highest mean utility given price; County Line and the "other national" group the
   lowest. Small packages are least preferred; specialty cheese has the highest type effect.

### 7.4 Smoke test of the benchmark (run 7, 175 markets)

Run time 0.2 minutes; both GMM steps converge.

| Parameter | Estimate | SE |
|---|---|---|
| price (mean $\alpha$) | −10.66 | 4.97 |
| $\sigma$ constant | 0.000 | 34.1 |
| $\sigma$ price | 0.000 | 9.39 |
| $\sigma$ family size | 0.000 | 1.65 |
| $\pi$ constant × income | 0.207 | 53.6 |
| $\pi$ price × income | −0.459 | 3.82 |
| $\pi$ family size × household size | **0.329** | **0.097** |

Own-price elasticities (5th, 25th, 50th, 75th, 95th percentiles): −4.32, −2.99, −2.38, −1.99,
−1.50. No product-market has a positive own-price elasticity.

Family size × household size is already positive and significant here (t ≈ 3.4). All three σ
sit at their lower bound of 0 in this small sample; with more markets, $\sigma_{\text{price}}$
and $\sigma_{\text{family size}}$ move away from 0 and become significant (§7.1).

---

## 8. Promotion controls

### 8.1 Motivation

Dominick's flags each UPC-store-week with a deal code. Two codes matter for cheese: **B**
(bonus buy) and **S** (simple price reduction). The question here is whether consumers respond
to the *information* that a product is on promotion (shelf tags, signage, display), over and
above the price cut itself. The price cut is already captured by $p_{jt}$, so a promotion
variable in $X_1$ measures the extra utility from the promotion signal at a given price.

Promotions are observed at the store-week level, not for each shopper, so they enter as
product-market characteristics in the mean utility $\delta_{jt}$. They are treated as
exogenous and therefore also serve as their own instruments.

Note that the Dominick's code B means "bonus buy", an in-store featured deal; it is not
necessarily a buy-one-get-one offer.

### 8.2 Measures

A product merges several UPCs, so a promotion can cover part of a product. Three measures,
each with one column for B and one for S:

| Option | Definition | Interpretation |
|---|---|---|
| `--promo any` | 1 if any UPC of the product carries the code | a promotion tag is visible for the product |
| `--promo fw` | $\sum_u w_{us}\,1[u \text{ on deal}] / \sum_u w_{us}$ over the product's UPCs sold that week; $w_{us}$ is UPC $u$'s share of the product's units in store $s$ over the whole sample | probability that a typical shopper of the product meets a promoted UPC |
| `--promo fwdummy` | $1[\texttt{fw} \ge 0.5]$ (`--promo-cut`) | most of the product is on promotion |

**Why fixed weights.** The obvious measure, the share of this week's units sold on the deal
(`promo_B_share`, `promo_S_share`, available as `--promo share`), has this week's quantity in
the denominator. A promotion raises the promoted UPC's sales, so that share moves mechanically
with the demand shock $\xi_{jt}$. The fixed weights $w_{us}$ do not change from week to week,
so `fw` reflects only *which* UPCs are promoted. The two measures are highly correlated (0.97
for B, 0.96 for S), but only `fw` can be treated as an exogenous characteristic.

**Descriptive statistics** (estimation sample, 1,087,156 product-markets):

| | B | S |
|---|---|---|
| product-markets with any UPC on the deal | 28.0% | 5.0% |
| mean of `fw` | 0.148 | 0.028 |
| mean of `fw` when positive | 0.53 | 0.57 |
| share with `fw` ≥ 0.5 | 13.9% | 2.8% |
| log price relative to product-store mean, on vs off the deal | −0.048 vs +0.019 | −0.084 vs +0.004 |
| correlation of `fw` with that log price deviation | −0.30 | −0.19 |

B and S rarely coincide (1.1% of product-markets). Random-weight cheese is never on B or S, and
small packages rarely are. Promotions coincide with lower prices, but far from perfectly.

**Are price and `fw` collinear?** No. Promotions lower prices, but once the store and week
fixed effects are absorbed and the 19 dummies partialled out (the space in which pyBLP
estimates $\alpha$), the promotion measures explain little of the price variation:

| Check (full estimation sample, after FE and dummies) | Value |
|---|---|
| correlation of price with B `fw` / S `fw` | −0.13 / −0.11 |
| $R^2$ of price on (B `fw`, S `fw`) | 0.032 |
| variance inflation factor of price | 1.03 |
| share of price variation left after the promotion controls | 96.8% |
| correlation of own wholesale cost with B `fw` / S `fw` | −0.05 / −0.04 |
| partial $R^2$ of own wholesale cost for price, without / with promotion controls | 0.846 / 0.856 |

Most price variation comes from wholesale cost, differences between products and regular price
changes, not from promotions. `fw` measures how much of a product is promoted, not the depth
of the discount, so it is only loosely tied to the volume-weighted price. The cost instrument is
nearly uncorrelated with `fw`, and the first stage is unaffected. Consistent with this, the
standard error of $\alpha$ barely changes when the controls are added (0.45 → 0.46, §8.3).
Price and the promotion signal are therefore separately identified. Script:
[code/diagnostics/promo_collinearity.py](code/diagnostics/promo_collinearity.py).

### 8.3 Results

All other settings are the benchmark defaults (§1). Standard errors in parentheses.

**Quarterly subsample (2,170 markets, 46 instruments)**

| | none (benchmark) | any | fw | fwdummy |
|---|---|---|---|---|
| price (mean $\alpha$) | −7.46 (1.52) | −4.62 (1.58) | −4.54 (1.63) | −5.05 (1.60) |
| promo B | — | 0.45 (0.02) | 0.47 (0.03) | 0.31 (0.02) |
| promo S | — | 0.76 (0.03) | 1.10 (0.06) | 0.89 (0.04) |
| $\sigma$ price | 2.65 (0.87) | 1.93 (0.89) | 1.89 (0.92) | 2.05 (0.90) |
| $\sigma$ family size | 0.32 (0.34) | 0.36 (0.44) | 0.32 (0.45) | 0.32 (0.42) |
| $\pi$ price × income | −0.20 (0.27) | −0.10 (0.25) | −0.06 (0.25) | −0.12 (0.25) |
| $\pi$ family size × household size | 0.38 (0.03) | 0.38 (0.03) | 0.38 (0.03) | 0.38 (0.03) |
| median own elasticity | −1.40 | −0.88 | −0.87 | −0.96 |
| own elasticity, 5th–95th pct. | −2.26 to −0.93 | −1.46 to −0.57 | −1.43 to −0.56 | −1.57 to −0.62 |

**Full sample, `--promo fw` (27,230 markets, 49 instruments)**

| | Benchmark (§7) | **With `fw` promotion controls** |
|---|---|---|
| price (mean $\alpha$) | −8.21 (0.45) | **−6.19 (0.46)** |
| promo B (fw) | — | **0.46 (0.009)** |
| promo S (fw) | — | **0.98 (0.016)** |
| $\sigma$ constant | 0.00 (0.21) | 0.00 (0.23) |
| $\sigma$ price | 2.93 (0.24) | 2.50 (0.24) |
| $\sigma$ family size | 0.33 (0.10) | 0.35 (0.11) |
| $\pi$ constant × income | 0.31 (0.25) | 0.12 (0.25) |
| $\pi$ price × income | −0.12 (0.08) | −0.08 (0.07) |
| $\pi$ family size × household size | 0.36 (0.008) | 0.36 (0.009) |
| own elasticity: 5th / 25th / **50th** / 75th / 95th pct. | −2.41 / −1.80 / **−1.50** / −1.28 / −1.00 | −1.83 / −1.36 / **−1.13** / −0.96 / −0.75 |
| products with median own elasticity above −1 | 1 of 41 | 11 of 41 |
| run time / peak memory (Longleaf, one core) | 64 min / 37 GB | 101 min / 40 GB |

Command: `python code/03_blp_nevo.py --every 1 --promo fw`. Both GMM steps converge (30
optimizer iterations in total); the step-2 projected gradient norm is 0.05 and the
weighting-matrix condition number $2.1\times10^6$. As in the benchmark, the reduced Hessian has
one zero eigenvalue, from $\sigma_{\text{constant}}$ at its bound. The linear characteristics
barely move, except that the type effects of grated parmesan, specialty and snacks (high
prices per ounce) fall by 0.15–0.27 and that of cream cheese rises by 0.15.

### 8.4 Interpretation

1. **The promotion signal matters.** Both coefficients are large and precisely estimated in
   every measure. In dollar terms (coefficient ÷ |α|, full sample): an S tag on the whole
   product is worth $0.16/oz, about 65% of the mean price; a B promotion is worth $0.07/oz,
   about 30%. Simple price reductions carry the stronger signal.
2. **Price sensitivity was overstated without the controls.** Without them, the sales spike
   in promotion weeks is attributed entirely to the price cut. With them, the mean price
   coefficient falls by a quarter in the full sample (−8.2 → −6.2) and by about 40% in the
   quarterly subsample; $\sigma_{\text{price}}$ falls from 2.9 to 2.5 (0.7% of simulated
   consumers then have a positive price coefficient).
3. **Heterogeneity is unchanged.** Family size × household size stays at 0.36; price × income
   and constant × income remain insignificant.
4. **The three measures agree.** `any` and `fw` give almost identical results; `fwdummy` gives
   a somewhat smaller B coefficient and a larger price coefficient, because it codes partial
   promotions as zero. `fw` is the preferred measure: it matches the "probability that a
   shopper sees the promotion" interpretation and does not move with this week's quantities.
5. **Full sample vs quarterly.** The price coefficient is larger in magnitude in the full
   sample (−6.2 vs −4.5), whereas without promotion controls the two samples agreed closely
   (−8.2 vs −7.5). With promotion controls, price is identified only from cost variation not
   accompanied by a promotion; a likely reason is that this variation is thinner in the
   quarterly subsample. The full-sample estimates are the ones to report.

**Caveats.**

- **Elasticities near or below one in magnitude.** The median own elasticity is −1.13 and 11 of
  41 products have a median above −1 (cream cheese, processed slices, Velveeta). With a 34%
  average retail margin, a single-product price setter would imply an elasticity around −3;
  inelastic demand is inconsistent with profit maximization on that product alone.
  Multiproduct pricing and loss-leader pricing soften this. Mechanical collinearity between
  price and the promotion controls is ruled out (§8.2), so the fall in $|\alpha|$ reflects
  the removal of omitted-variable bias: without the controls, the promotion signal sits in
  $\xi_{jt}$ and its demand effect is attributed to price. More likely sources of the low
  elasticities are the product definition (volume elasticities of coarse cells are lower than
  UPC elasticities, already visible in the benchmark) and promotion endogeneity (below).
- **Instrument interaction.** Manufacturer trade deals could lower wholesale cost and trigger
  promotions at the same time. In practice the cost instrument is almost uncorrelated with
  `fw` (−0.05) and its first-stage strength is unchanged by the controls (§8.2), so this
  channel is small.
- **Exogeneity.** Promotion timing is a retailer decision and may respond to expected demand
  (e.g. holidays; week fixed effects absorb chain-wide timing). If promotions are scheduled in
  high-$\xi$ weeks, their coefficients are biased upward.

The benchmark (no controls, median −1.50) and this specification (median −1.13) bracket the
plausible own-price elasticities; reporting both, with the direction of each bias, is the
honest summary.

---

## 9. Running on Longleaf (UNC)

1. Copy code and data from the project root (only `code/`, `demo.dta` and `data/*.parquet`
   are needed):
   ```bash
   rsync -a --exclude __pycache__ code demo.dta longleaf:empirical_io_ps3/
   rsync -a data/panel_week_coarse.parquet longleaf:empirical_io_ps3/data/
   ```
2. One-time environment setup on a login node: `bash code/longleaf/setup.sh`
   (module `python/3.12.4`, pyBLP 1.3.0).
3. Submit:
   ```bash
   sbatch --partition=general --mem=48g --time=12:00:00 --job-name=BENCH_full \
          --export=ALL,FE=store_week,EVERY=1 code/longleaf/run_nevo.sl
   sbatch --partition=general --mem=16g --time=12:00:00 --job-name=BENCH_q13 \
          --export=ALL,FE=store_week,EVERY=13 code/longleaf/run_nevo.sl
   ```
   Extra estimation options are passed through `EXTRA`, e.g.
   `--export=ALL,FE=store_week,EVERY=13,EXTRA="--pi full"`.
4. Monitor with `squeue -u <onyen>` and `tail -f logs/<job>_<id>.out`.

**Resources.** Jobs run on one core: `pyblp.parallel` was much slower than a single process
on this problem, because there are many small markets and inter-process data transfer
dominates. Actual benchmark runs: quarterly subsample 5.5 minutes and 3.1 GB; full sample
64 minutes and 37 GB, at about 50 seconds per objective evaluation (step 2: 18 optimizer
iterations, 22 objective evaluations). `--mem=48g` on the `general` partition is enough for
the full sample.

To fetch results to a laptop (macOS ships an old rsync that accepts one remote path per call):
```bash
rsync -av --exclude '*.pkl' longleaf:empirical_io_ps3/output/ output/
scp 'longleaf:empirical_io_ps3/logs/BENCH_*' logs/
```
Off campus, connect to the UNC VPN first.

---

## 10. Remaining departures and extensions

1. **Market size.** $M_s$ is fixed per store and driven by one holiday week in most stores.
   Robustness: `--mw-quantile 0.99`, `--mw-mult 2`, or `--market-size traffic`.
2. **Nest.** The nesting parameter is not identified; fixing ρ at a few values (e.g. 0.3, 0.5,
   0.7) and comparing elasticities would follow Miller–Weinberg's robustness checks.
3. **Promotions.** The benchmark has no promotion controls, so the demand effect of the
   promotion signal loads on price. §8 adds them (`--promo fw`); a random coefficient on the
   promotion variables, interacted with income, would show which consumers respond to them.
4. **Optimal instruments.** `results.compute_optimal_instruments()` in pyBLP (Reynaert and
   Verboven, 2014) may sharpen the estimates of Σ and Π.
5. **Stockpiling.** A reduced-form check finds a post-promotion dip of about 8% in the
   following week (about 15–20% of the promotion lift), so static elasticities are short-run
   and likely somewhat overstated.
6. **Coarse products** mix UPCs within a cell (e.g. 12- and 16-oz Kraft Singles, regular and
   reduced-fat), so a promotion on one UPC moves the product's average price only partly.
