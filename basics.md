# Cheese demand, retail conduct and a ban on store-cut cheese: Dominick's, 1989–1997

This note is the short version of the project. It has three parts: the demand estimates
(Section 2), what they imply about vertical conduct (Section 3), and a counterfactual that
removes random-weight private-label cheese (Section 4). Details, robustness checks and the
specification history are in `demand_estimation_nevo_style.md`.

**Main findings**

1. BLP instruments are weak in this panel (first-stage partial R² of 0.006) and their
   estimates move with the sample and the instrument subset (price coefficient −16 to −30).
   Adding wholesale cost gives a stable price coefficient of −8.2 (0.45) and a median
   own-price elasticity of −1.5. Promotions shift demand strongly (bonus buy 0.46, price
   reduction 0.98 in utility units).
2. Among the conduct models, product-by-product retail pricing fits the observed retail
   margins best. Category-wide retail pricing, manufacturer price setting and double
   marginalization are rejected; double marginalization implies negative manufacturing cost
   for 77% of observations. Every model over-predicts the observed 34% margin.
3. Removing store-cut private-label cheese (14% of volume) lowers cheese volume by about 10%,
   consumer surplus by about 11% and the retailer's gross margin by 11–13%. Only 27–31% of
   the lost volume moves to packaged cheese, and remaining prices rise by less than 1%.

---

## 1. Setup

| Item | Choice |
|---|---|
| Data | Dominick's cheese category, 83 stores with demographics, 380 weeks |
| Market | store × week: 27,230 markets, 1,087,156 product-market observations |
| Products | 41 brand group × type group × size class cells (90% of units) |
| Shares | 8-oz equivalent units / market size; market size = 1.5 × the store's maximum weekly sales |
| Price | $ per ounce, volume-weighted over the product's UPCs |
| Linear part | price, 4 size, 7 brand and 8 type dummies; store and week fixed effects absorbed |
| Random coefficients | constant, price, family-size package (diagonal Σ); constant × income, price × income, family size × household size |
| Consumers | 20 per store: log income from the store's lognormal distribution, household size at the store mean |
| BLP instruments | sums of size-class and type-group dummies over the firm's other products and over rivals' products (28), store demographics × characteristics (20) |
| Cost instruments | own wholesale cost, mean wholesale cost of rival products in the market and in the type group (3) |
| Promotion controls | fixed-weight share of the product's UPCs on bonus buy (B) and on simple price reduction (S) |

Code: `code/03_blp_nevo.py` (demand), `code/04_supply.py` (Sections 3 and 4).

---

## 2. Demand estimates

Three full-sample estimates. Columns (1) and (2) differ only in the instruments: (2) adds
the three wholesale-cost instruments to the BLP instruments of (1). Column (3) adds the
promotion controls to (2). Standard errors in parentheses.

| | (1) BLP instruments | (2) + wholesale cost (main) | (3) + wholesale cost, promotion controls |
|---|---|---|---|
| Price | −20.90 (0.51) | −8.21 (0.45) | −6.19 (0.46) |
| Promo B | | | 0.46 (0.01) |
| Promo S | | | 0.98 (0.02) |
| σ constant | 0.00 (0.58) | 0.00 (0.21) | 0.00 (0.23) |
| σ price | 1.62 (0.43) | 2.93 (0.24) | 2.50 (0.24) |
| σ family size | 0.67 (0.29) | 0.33 (0.10) | 0.35 (0.11) |
| constant × income | −0.01 (0.18) | 0.31 (0.25) | 0.12 (0.25) |
| price × income | −0.31 (0.12) | −0.12 (0.08) | −0.08 (0.07) |
| family size × household size | 0.34 (0.02) | 0.36 (0.01) | 0.36 (0.01) |
| Private label | 0.47 | 1.19 | 1.26 |
| Kraft | 1.03 | 0.99 | 1.00 |
| Philadelphia | 1.48 | 1.59 | 1.60 |
| Sargento | 2.04 | 1.68 | 1.66 |
| Random weight (size class) | 1.12 | 0.26 | 0.37 |
| Own-price elasticity, median | −4.54 | −1.50 | −1.13 |
| 5th / 95th percentile | −8.33 / −2.92 | −2.41 / −1.00 | −1.83 / −0.75 |
| Excluded instruments | 46 | 49 | 49 |

Brand and size coefficients are relative to Borden and the bulk size class. Parameter files:
`output/nevo_params_coarse_every1_*.csv` (tags `pi-slim_demoiv`, `pi-slim_demoiv_costiv`,
`pi-slim_demoiv_costiv_promo-fw`).

### 2.1 BLP instruments are weak here

First stage for price, after store and week fixed effects and the 19 dummies (quarterly
subsample):

| Excluded instruments | Partial R² | F |
|---|---|---|
| BLP sums | 0.006 | 23 |
| Demographics × characteristics | 0.001 | 4.6 |
| Own wholesale cost | 0.835 | 438,441 |
| All | 0.839 | 9,775 |

- **Why.** Product characteristics do not vary over time, so BLP sums move only when the
  choice set changes. The 41 products are sold in almost every store-week (standard
  deviation of the number of products per market: 0.7), so the sums are close to constant
  within a product once the fixed effects and dummies are removed.
- **The estimates are not stable.** With the same specification the price coefficient is
  −15.6 (1.6) on the quarterly subsample and −20.9 (0.5) on the full sample. Dropping the
  demographic instruments and keeping only the BLP sums gives −29.6 (2.4), with σ price at
  zero and demographic interactions of 20 and −15. With wholesale cost the two samples give
  −7.5 (1.5) and −8.2 (0.45).
- **The standard errors in column (1) should not be trusted.** They are small, but
  conventional GMM standard errors are not valid with weak instruments, and the instability
  above is the better guide to precision.
- **The two instrument sets disagree on the level of the elasticity**, and this matters for
  Section 3. Column (1) gives a median elasticity of −4.5, column (2) gives −1.5. The
  retailer's observed margin of 34% corresponds to an elasticity near −3 under
  product-by-product pricing, between the two. Wholesale cost is far stronger, but it is
  not beyond doubt: it is an average acquisition cost, and trade deals can move it together
  with promotions (Section 5). Column (2) is the main specification because its estimates
  are stable and its first stage is strong, not because column (1) is shown to be wrong.

### 2.2 Promotion controls

- Both promotion coefficients are large and precisely estimated. A simple price reduction
  tag is worth about twice a bonus buy.
- Adding the controls lowers the price coefficient in absolute value (−8.2 to −6.2), and
  the median elasticity from −1.5 to −1.1: part of what looked like a price response is a
  response to the promotion signal.
- Column (2) stays the main specification for the supply side. In column (3) about 30% of
  product-markets have inelastic demand, and promotions are a retailer choice that the
  supply models below do not describe.

### 2.3 Other remarks

- **σ constant** is at its lower bound of zero in all columns: store and week fixed effects
  leave little variation in the inside share to identify it.
- **Optimal instruments** (Chamberlain 1987; Reynaert and Verboven 2014) computed from
  column (2) halve the standard errors but give a price coefficient of −4.6 and a median
  elasticity of −0.82, which no profit-maximizing conduct can rationalize. They are not
  used; see `demand_estimation_nevo_style.md`.

---

## 3. Vertical conduct

### 3.1 Approach

Each store-week has one retailer, and Dominick's records its wholesale cost $w$ for every
UPC. So the retail margin $p - w$ is observed (median 34% of price), and the retail side of
each conduct model can be compared with data directly. The models follow Villas-Boas (2007):

| Model | Who sets what | Total margin $p - c$ |
|---|---|---|
| R-prod | Retailer prices each product on its own, given $w$ | $-s_j / (\partial s_j / \partial p_j)$ |
| R-brand | Retailer prices brand by brand, given $w$ | Bertrand, ownership = brand group |
| R-cat | Retailer maximizes category profit, given $w$. With nonlinear wholesale contracts ($w$ = manufacturer marginal cost) this is the vertically integrated outcome | $-(\partial s / \partial p')^{-1} s$ |
| M-bert | Manufacturers set retail prices; retail margin is zero; the retailer owns the private label | Bertrand, ownership = manufacturer |
| DM | Double marginalization: manufacturers set linear wholesale prices, retailer then maximizes category profit | R-cat retail margin + manufacturer margin through retail pass-through |

For R-prod, R-brand and R-cat the model margin is a prediction of the observed $p - w$.
Under M-bert the predicted retail margin is zero. Under DM the retail part is the R-cat
margin, and the manufacturer margin is $-(T_w \circ (\Delta_p G)')^{-1} s$, with $G$ the
pass-through matrix from the retailer's first-order conditions (it uses the second
derivatives of demand). Private-label wholesale prices are treated as non-strategic.

Mean utilities are recovered by share inversion at the main estimates (column 2 of Section
2); the implied elasticities match pyBLP's to eight digits.

### 3.2 Results

| | Median margin / p | Quantity-weighted margin / p | Implied mc < 0 | RMSE vs observed retail margin (share of p) |
|---|---|---|---|---|
| Observed retail margin | 0.34 | 0.30 | | |
| R-prod | 0.67 | 0.83 | 5% | 0.41 |
| R-brand | 0.71 | 0.88 | 8% | 0.46 |
| M-bert | 0.73 | 0.93 | 12% | 0.49 |
| R-cat | 0.92 | 1.17 | 38% | 0.69 |
| DM | 1.26 | 1.70 | 77% | |

Pairwise comparison of fit to the observed retail margin (difference in squared errors,
standard errors clustered by store; negative t: the first model fits better):

| R-prod vs R-brand | R-prod vs R-cat | R-brand vs R-cat | M-bert vs R-prod |
|---|---|---|---|
| t = −25.5 | t = −25.2 | t = −24.9 | t = +28.3 |

Under R-cat the median pass-through of own wholesale cost to the retail price is 1.25,
close to the 1.27 in the raw data. The DM manufacturer margin on national brands is 54% of
the retail price at the median.

By product group (quantity-weighted):

| | National brands | Private label, packaged | Private label, random weight |
|---|---|---|---|
| Share of volume | 61% | 25% | 14% |
| Price, $/oz | 0.204 | 0.174 | 0.187 |
| Observed retail margin / p | 0.24 | 0.37 | 0.40 |
| Median own elasticity | −1.59 | −1.39 | −1.30 |
| R-prod margin / p | 0.82 | 0.88 | 0.78 |
| R-cat margin / p | 1.16 | 1.24 | 1.11 |

### 3.3 What the test says

- **The model with the least internalization fits best.** Implied margins rise in the order
  R-prod < R-brand < M-bert < R-cat < DM, and all exceed the observed 34%. A
  Miller–Weinberg style internalization parameter κ (ownership matrix with κ off the
  diagonal) would be estimated at its lower bound of zero.
- **Double marginalization is rejected.** It needs a manufacturer margin of 54% of the
  retail price on top of the retail margin, which implies negative manufacturing cost for
  77% of observations.
- **Category-wide retail pricing is rejected** against product-by-product pricing: it
  implies margins near 92% of price and negative cost in 38% of observations. Dominick's
  behaves like a retailer that takes wholesale prices as given and does not fully exploit
  category-level market power.
- **Manufacturer retail-price setting is rejected directly**: it predicts a zero retail
  margin, and the observed margin is 34% (24% on national brands, 37–40% on private label).
- **Even the best model over-predicts**: R-prod implies 67% against 34% observed, which
  would need a median own elasticity near −3. Likely reasons: products are coarse
  aggregates (substitution within a cell is not counted), weekly data mix promotion timing
  with long-run response, and competition from other chains is outside a store-level
  market. The private-label premium in observed margins (37–40% vs 24%) is not reproduced
  either, which points to wholesale prices of national brands being above marginal cost.

Conduct used in Section 4: R-prod.

---

## 4. Counterfactual: packaged cheese only

### 4.1 Design

Suppose store-cut, random-weight cheese can no longer be sold (for example for hygiene
reasons). In the data this is four private-label products: cheddar, other natural,
mozzarella and processed, sold by weight. They are present in every market and account for
14.0% of inside volume, at an observed margin of 40%.

The four products are removed from every choice set. Two versions:

- **Prices fixed.** Uses demand only.
- **Re-priced.** Remaining prices are re-solved under R-prod, holding fixed the marginal
  cost that R-prod implies at observed prices.

Consumer surplus is the expected maximum utility divided by each consumer's marginal utility
of a dollar. Retail gross margin is $(p - w) q$ at observed wholesale costs. Baseline per
store-week: revenue 6,216 dollars, gross margin 1,948 dollars.

### 4.2 Results

| | Prices fixed | Re-priced (R-prod) |
|---|---|---|
| Cheese volume | −9.7% | −10.3% |
| Lost volume diverted to remaining cheese | 31% | 27% |
| Remaining private label: price | 0 | +0.7% |
| Remaining national brands: price | 0 | +0.9% |
| Remaining private label: quantity | +4.9% | +4.3% |
| Remaining national brands: quantity | +5.1% | +4.3% |
| Consumer surplus | −10.7% | −11.2% |
| Consumer surplus, $ per store-week | −869 | −916 |
| Retail gross margin | −12.8% | −11.2% |
| Retail gross margin, $ per store-week | −249 | −218 |
| Variable profit at R-prod costs | −9.3% | −9.1% |

The pricing equilibrium was found in 95.2% of markets; the re-priced column uses those.

### 4.3 Interpretation

- **Volume.** About 70% of the removed volume leaves the category. Substitution to packaged
  cheese is spread almost evenly: packaged private label and national brands both gain
  4–5%, and packaged products of the same types gain no more than other types. This is the
  logit pattern. The random coefficients are on the constant (at zero), price and family
  size, so the model has no dimension along which store-cut cheddar is closer to packaged
  cheddar than to cream cheese. Diversion to close substitutes is probably understated and
  the volume loss overstated.
- **Prices.** Remaining prices rise by less than 1%. The effect is small because under
  R-prod the retailer did not internalize substitution between products to begin with;
  removing products only makes the remaining demand slightly less elastic.
- **Consumers.** Surplus falls by about 11%, or roughly 900 dollars per store-week against
  category revenue of 6,216 dollars. The figure includes the loss of the logit taste draw for four
  products, which tends to overstate the value of variety.
- **Retailer.** Gross margin falls by 13% at fixed prices: the removed products carry the
  highest margins (40%), and the diverted volume goes partly to national brands at 24%.
  Re-pricing recovers about one eighth of the loss.
- **Manufacturers.** National-brand volume rises by 4–5%, so branded manufacturers gain.
  Their profit change is not reported in dollars because their costs are not identified
  without a conduct model that fits.

---

## 5. Limitations

- All conduct models over-predict the retail margin, so the marginal costs used in the
  re-priced counterfactual are too low. The fixed-price column does not depend on this.
- Wholesale cost in Dominick's data is the average acquisition cost of inventory, not the
  replacement cost, so observed margins lag wholesale price changes.
- The conduct comparison is on retail margins only. Manufacturer marginal costs are not
  observed, so the upstream part of each model is judged by whether implied costs are
  positive, not by a formal cost-side test as in Villas-Boas (2007).
- The counterfactual holds wholesale prices, the product set of packaged cheese and
  store traffic fixed.

## References

- Chamberlain (1987), Asymptotic efficiency in estimation with conditional moment restrictions, *Journal of Econometrics*.
- Conlon and Gortmaker (2020), Best practices for differentiated products demand estimation with PyBLP, *RAND Journal of Economics*.
- Miller and Weinberg (2017), Understanding the price effects of the MillerCoors joint venture, *Econometrica*.
- Nevo (2000), A practitioner's guide to estimation of random-coefficients logit models of demand, *Journal of Economics & Management Strategy*.
- Reynaert and Verboven (2014), Improving the performance of random coefficients demand models: the role of optimal instruments, *Journal of Econometrics*.
- Villas-Boas (2007), Vertical relationships between manufacturers and retailers: inference with limited data, *Review of Economic Studies*.
