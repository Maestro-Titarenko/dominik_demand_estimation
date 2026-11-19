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

## 1. Model and data

### 1.1 Data

| Item | Choice |
|---|---|
| Data | Dominick's cheese category, 83 stores with demographics, 380 weeks |
| Market | store × week: 27,230 markets, 1,087,156 product-market observations |
| Products | 41 brand group × type group × size class cells (90% of units) |
| Quantity | 8-oz equivalent units |
| Price | dollars per ounce, volume-weighted over the product's UPCs |
| Wholesale cost | dollars per ounce, from Dominick's recorded gross margin |
| Promotion | fixed-weight share of the product's UPCs on bonus buy (B) and on simple price reduction (S) |

### 1.2 Utility

A market $t = (s, w)$ is a store $s$ in a week $w$. Consumer $i$ in market $t$ buys one
8-oz unit of one of the $J_t$ products or the outside good. The utility from product $j$ is

$$
u_{ijt} = x_j'\beta + \alpha_i\, p_{jt} + \gamma_i + \phi_i\, f_j + \xi_s + \xi_w + \xi_{jt} + \varepsilon_{ijt},
\qquad u_{i0t} = \varepsilon_{i0t},
$$

where

- $p_{jt}$ is the price and $x_j$ collects the size-class, brand-group and type-group
  dummies (19 after dropping base categories);
- $f_j = 1$ for family-size packages (12 oz and above);
- $\xi_s$ and $\xi_w$ are store and week fixed effects, and $\xi_{jt}$ is the demand shock
  that consumers and the retailer observe but the econometrician does not;
- $\varepsilon_{ijt}$ is i.i.d. type I extreme value.

In column (3) of Section 2 the two promotion variables are added to $x$ and vary by
product-market.

### 1.3 Random coefficients

Tastes for the inside goods, for price and for family-size packages vary across consumers:

$$
\begin{pmatrix} \gamma_i \\ \alpha_i \\ \phi_i \end{pmatrix}
=
\begin{pmatrix} 0 \\ \alpha \\ 0 \end{pmatrix}
+ \Sigma\, \nu_i + \Pi\, D_i ,
\qquad
\Sigma = \begin{pmatrix} \sigma_c & 0 & 0 \\ 0 & \sigma_p & 0 \\ 0 & 0 & \sigma_f \end{pmatrix},
\qquad
\Pi = \begin{pmatrix} \pi_c & 0 \\ \pi_p & 0 \\ 0 & \pi_f \end{pmatrix}.
$$

- $\nu_i \sim N(0, I_3)$ are unobserved tastes.
- $D_i = (\text{income}_i, \text{hhsize}_i)'$: log income is drawn from the store's
  lognormal income distribution, household size is the store mean. Both are demeaned.
- The mean tastes for the constant and for family size are absorbed in the fixed effects
  and in $x_j'\beta$.

The nonlinear parameters are $\theta = (\sigma_c, \sigma_p, \sigma_f, \pi_c, \pi_p, \pi_f)$;
$\alpha$ and $\beta$ enter the mean utility linearly.

Utility separates into a mean part and a consumer-specific part:

$$
u_{ijt} = \delta_{jt} + \mu_{ijt} + \varepsilon_{ijt},
\qquad
\delta_{jt} = x_j'\beta + \alpha\, p_{jt} + \xi_s + \xi_w + \xi_{jt},
$$

$$
\mu_{ijt} = (\sigma_c \nu_{i1} + \pi_c\, \text{income}_i)
+ (\sigma_p \nu_{i2} + \pi_p\, \text{income}_i)\, p_{jt}
+ (\sigma_f \nu_{i3} + \pi_f\, \text{hhsize}_i)\, f_j .
$$

### 1.4 Choice probabilities and market shares

The probability that consumer $i$ chooses product $j$ is

$$
s_{ijt} = \frac{\exp(\delta_{jt} + \mu_{ijt})}{1 + \sum_{k=1}^{J_t} \exp(\delta_{kt} + \mu_{ikt})} ,
$$

and the market share integrates over consumers. With $R = 20$ simulated consumers per
store, reused in every week,

$$
s_{jt}(\delta_t; \theta) = \frac{1}{R} \sum_{i=1}^{R} s_{ijt} .
$$

The observed share is $s_{jt} = q_{jt} / M_s$, where $q_{jt}$ is the number of 8-oz units
sold and the market size $M_s$ is 1.5 times the store's largest weekly cheese volume
(Miller and Weinberg 2017).

Price derivatives, used for the elasticities and the supply side:

$$
\frac{\partial s_{jt}}{\partial p_{kt}} = \frac{1}{R} \sum_{i} \alpha_i\, s_{ijt} \left( 1\{j = k\} - s_{ikt} \right),
\qquad
\eta_{jkt} = \frac{\partial s_{jt}}{\partial p_{kt}} \frac{p_{kt}}{s_{jt}} .
$$

### 1.5 Estimation

For given $\theta$, the mean utilities $\delta_t(\theta)$ solve
$s_{jt}(\delta_t; \theta) = s_{jt}$ for every product (Berry, Levinsohn and Pakes 1995
contraction). The demand shock is the residual

$$
\xi_{jt}(\theta) = \delta_{jt}(\theta) - x_j'\beta - \alpha\, p_{jt} - \xi_s - \xi_w ,
$$

and the parameters are estimated by two-step GMM on $E[\xi_{jt} Z_{jt}] = 0$ with pyBLP
(Conlon and Gortmaker 2020); the fixed effects are absorbed. Price is endogenous because
the retailer sets it knowing $\xi_{jt}$. The excluded instruments $Z_{jt}$ are:

| Group | Count | Construction |
|---|---|---|
| BLP sums | 28 | sums of size-class and type-group dummies over the firm's other products and over rival firms' products |
| Demographic | 20 | store mean demographics × size-class dummies and × number of rival products |
| Wholesale cost | 3 | own wholesale cost; mean wholesale cost of rival products in the market and in the type group |

Columns that are collinear after absorbing the fixed effects and partialling out $x_j$ are
dropped, which leaves 46 instruments without wholesale cost and 49 with it.

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

### 3.1 Setting

Each store-week has one retailer, and Dominick's records its wholesale cost $w_{jt}$ for
every product. So the retail margin $p_{jt} - w_{jt}$ is observed (median 34% of price),
and the retail side of each conduct model can be compared with data directly. The models
follow Villas-Boas (2007). Market subscripts are dropped below.

Notation: $s(p)$ is the vector of shares, $M$ the market size, and

$$
\Delta_{jk} = \frac{\partial s_k}{\partial p_j}
$$

the matrix of share derivatives from Section 1.4. $c_j$ is the marginal cost of the firm
that sets the price in question, and $\circ$ is the element-by-element product. An
ownership matrix $T$ has $T_{jk} = 1$ if products $j$ and $k$ are priced by the same
decision-maker and 0 otherwise.

### 3.2 Conduct models

**Retail pricing given wholesale prices (R-prod, R-brand, R-cat).** The retailer takes $w$
as its marginal cost. Products are split into pricing units; the unit $B$ that contains
product $j$ chooses its prices to maximize its own profit,

$$
\max_{\{p_k\}_{k \in B}} \; M \sum_{k \in B} (p_k - w_k)\, s_k(p) .
$$

The first-order condition for $p_j$ is

$$
s_j + \sum_{k \in B} (p_k - w_k) \frac{\partial s_k}{\partial p_j} = 0 ,
$$

and stacking over products gives the retail margin

$$
p - w = -\left( T_r \circ \Delta \right)^{-1} s .
$$

The three retail models differ only in $T_r$:

- **R-prod**: each product is its own unit, $T_r = I$, so
  $p_j - w_j = -s_j / (\partial s_j / \partial p_j)$. The retailer ignores substitution
  between products.
- **R-brand**: a unit is a brand group (brand managers); $T_r$ is block-diagonal by brand.
- **R-cat**: one unit, $T_r$ is a matrix of ones. The retailer maximizes category profit
  and internalizes all substitution.

**Vertical integration / nonlinear contracts.** If manufacturers sell at marginal cost and
collect fixed fees (two-part tariffs), the retailer is the residual claimant and maximizes
joint profit $M \sum_k (p_k - c_k) s_k(p)$. The total margin is the R-cat expression with
$c$ in place of $w$:

$$
p - c = -\Delta^{-1} s .
$$

With a single retailer per market this is also the monopoly outcome. It has the same
margin as R-cat and is not listed separately.

**Manufacturer pricing, zero retail margin (M-bert).** Manufacturers set retail prices
directly and the retailer earns no margin on their products. Manufacturer $f$ with product
set $F_f$ solves

$$
\max_{\{p_k\}_{k \in F_f}} \; M \sum_{k \in F_f} (p_k - c_k)\, s_k(p)
\quad\Longrightarrow\quad
p - c = -\left( T_m \circ \Delta \right)^{-1} s ,
$$

with $T_m$ the manufacturer ownership matrix (Kraft owns Kraft, Philadelphia and Velveeta;
the retailer owns the private label). Manufacturers compete Bertrand-Nash. The model
predicts $p - w = 0$ for national brands.

**Double marginalization (DM).** Manufacturers move first and set linear wholesale prices;
the retailer then sets retail prices.

*Stage 2.* The retailer maximizes category profit given $w$, as in R-cat:

$$
s(p) + \Delta\, (p - w) = 0 . \tag{1}
$$

This defines retail prices as a function of wholesale prices, $p(w)$.

*Stage 1.* Manufacturer $f$ anticipates $p(w)$ and solves

$$
\max_{\{w_k\}_{k \in F_f}} \; M \sum_{k \in F_f} (w_k - c_k)\, s_k\big(p(w)\big) ,
$$

with first-order condition

$$
s_j + \sum_{k \in F_f} (w_k - c_k) \frac{\partial s_k}{\partial w_j} = 0 ,
\qquad
\frac{\partial s_k}{\partial w_j} = \sum_{l} \frac{\partial s_k}{\partial p_l} \frac{\partial p_l}{\partial w_j} .
$$

Let $G$ be the pass-through matrix, $G_{lj} = \partial p_l / \partial w_j$, so that
$\Delta_w = G'\Delta$ has typical element $\partial s_k / \partial w_j$. The manufacturer
margin is

$$
w - c = -\left( T_m \circ \Delta_w \right)^{-1} s ,
$$

and the total margin is the sum of the two layers,

$$
p - c = -\Delta^{-1} s - \left( T_m \circ \Delta_w \right)^{-1} s .
$$

The pass-through matrix comes from totally differentiating (1) with respect to $w$:

$$
G = \left( \Delta + \Delta' + H \right)^{-1} \Delta ,
\qquad
H_{jl} = \sum_{k} (p_k - w_k) \frac{\partial^2 s_k}{\partial p_j\, \partial p_l} ,
$$

where the second derivatives are computed analytically from the choice probabilities and
$p - w$ is the R-cat margin. Private-label wholesale prices are not strategic: the
private label has no manufacturer margin, and its rows and columns are left out of the
stage-1 system.

### 3.3 Comparing the models

Mean utilities are recovered by share inversion at the main estimates (column 2 of Section
2), which gives $\Delta$ in every market; the implied elasticities match pyBLP's to eight
digits. Each model then delivers a margin vector $m^h$ and an implied marginal cost
$c^h = p - m^h$. Three checks:

1. **Retail margin.** For R-prod, R-brand and R-cat, $m^h$ is a prediction of the observed
   $p - w$; M-bert predicts zero. Fit is measured by the squared error
   $((m^h_{jt} - (p_{jt} - w_{jt})) / p_{jt})^2$, and two models are compared by the mean
   difference in squared errors with standard errors clustered by store (in the spirit of
   Rivers and Vuong 2002).
2. **Sign of implied cost.** A model is implausible if $c^h < 0$ for a large share of
   observations.
3. **Level.** The implied Lerner index $m^h / p$ against the observed retail margin.

### 3.4 Results

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

### 3.5 What the test says

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

Consumer surplus per market is the expected maximum utility converted to dollars. Price is
in dollars per ounce and a unit is 8 oz, so the marginal utility of a dollar is
$-\alpha_i / 8$:

$$
CS_t = M_s\, \frac{1}{R} \sum_i \frac{8}{-\alpha_i} \ln\Big( 1 + \sum_{j} \exp(\delta_{jt} + \mu_{ijt}) \Big) .
$$

Simulated consumers with $\alpha_i \ge 0$ (0.2%) are left out. Retail gross margin is
$8 M_s \sum_j (p_{jt} - w_{jt}) s_{jt}$ at observed wholesale costs. Baseline per
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

- Berry, Levinsohn and Pakes (1995), Automobile prices in market equilibrium, *Econometrica*.
- Chamberlain (1987), Asymptotic efficiency in estimation with conditional moment restrictions, *Journal of Econometrics*.
- Conlon and Gortmaker (2020), Best practices for differentiated products demand estimation with PyBLP, *RAND Journal of Economics*.
- Miller and Weinberg (2017), Understanding the price effects of the MillerCoors joint venture, *Econometrica*.
- Nevo (2000), A practitioner's guide to estimation of random-coefficients logit models of demand, *Journal of Economics & Management Strategy*.
- Reynaert and Verboven (2014), Improving the performance of random coefficients demand models: the role of optimal instruments, *Journal of Econometrics*.
- Rivers and Vuong (2002), Model selection tests for nonlinear dynamic models, *Econometrics Journal*.
- Villas-Boas (2007), Vertical relationships between manufacturers and retailers: inference with limited data, *Review of Economic Studies*.
