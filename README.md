# Dominick's Cheese: Random-Coefficients Logit Demand

BLP demand estimation for packaged cheese at Dominick's Finer Foods (Chicago, 1989–1997).
The model follows Nevo (2000); product definition, quantity units and market size follow
Miller and Weinberg (2017).

**Start with [demand_estimation_nevo_style.md](demand_estimation_nevo_style.md)**: full
specification, data construction, instruments, the specification history and the benchmark
results.

## Raw data (not included)

Place these Dominick's files in the project root:

| File | Content |
|---|---|
| `wche.csv` | weekly store × UPC movement for cheese (657 MB) |
| `upcche.csv` | cheese UPC descriptions |
| `ccount.dta` | daily store traffic (customer counts) |
| `demo.dta` | store trade-area demographics |

## Environment

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt   # benchmark versions
# or, on Python 3.9 (e.g. macOS system Python):
/usr/bin/python3 -m venv .venv && .venv/bin/pip install -r requirements-py39.txt
```

Both environments give the same estimates up to numerical noise. Key package: pyBLP 1.3.0.

## Pipeline

Run from the project root, in order:

| Step | Command | Output | Time |
|---|---|---|---|
| 1. UPC attributes | `python code/01_upc.py` | `data/upc_clean.csv` | seconds |
| 2. Panel (41 products) | `python code/02_clean.py --product-def coarse --coverage 0.90` | `data/panel_week_coarse.parquet`, `products_coarse.csv`, `markets_coarse.csv` | ~1 min |
| 2b. Panel (150 products, optional) | `python code/02_clean.py` | `data/panel_week.parquet`, … | ~1 min |
| 3. Smoke test | `python code/03_blp_nevo.py --every 100` | `output/nevo_*_every100_*` | < 1 min |
| 4. Benchmark | `python code/03_blp_nevo.py --every 1` | `output/nevo_*_every1_*` | ~1 h, ~37 GB RAM |

Step 4 was run on UNC Longleaf with `code/longleaf/run_nevo.sl`; setup instructions are in
section 9 of the documentation. All estimation options are listed in the header of
`code/03_blp_nevo.py` and with `python code/03_blp_nevo.py --help`.

## Layout

```
README.md                          this file
demand_estimation_nevo_style.md    documentation and results
requirements.txt                   pinned environment (Python 3.12, Longleaf)
requirements-py39.txt              pinned environment (Python 3.9, macOS)
code/
  01_upc.py                        parse UPC descriptions: brand, type, form, size
  02_clean.py                      clean movement and traffic data, build the panel
  03_blp_nevo.py                   pyBLP estimation
  groups.py                        brand groups (8) and type groups (9)
  diagnostics/                     scripts behind the numbers quoted in the documentation
    first_stage.py                   instrument strength (section 5.2)
    product_defs.py                  coverage of alternative product definitions (3.1)
    promo_stats.py                   promotion frequency and depth (3.5, 5.3)
    stockpile.py                     post-promotion dips (10)
    agg_elasticity.py                elasticities under time aggregation (10)
  longleaf/
    setup.sh                         one-time environment setup on Longleaf
    run_nevo.sl                      SLURM job script
output/                            benchmark estimates (full sample, quarterly, smoke test)
  archive/                         earlier exploratory runs, superseded
logs/                              Longleaf logs of the benchmark runs
```

## Benchmark at a glance (full sample, 27,230 store-weeks)

| | Estimate (SE) |
|---|---|
| price coefficient, mean | −8.21 (0.45) |
| σ price | 2.93 (0.24) |
| σ family-size package | 0.33 (0.10) |
| π family size × household size | 0.36 (0.01) |
| π price × income | −0.12 (0.08) |
| π constant × income | 0.31 (0.25) |
| median own-price elasticity | −1.50 (5th–95th pct: −2.41 to −1.00) |

Price is in dollars per ounce. Details and caveats: sections 7, 8 and 10 of the documentation.
