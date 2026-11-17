"""Parse upcche.csv: brand, product segment, cheese type, package size (oz).

Output: data/upc_clean.csv
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data"
OUT.mkdir(exist_ok=True)

u = pd.read_csv(ROOT / "upcche.csv")
u["DESCRIP"] = u["DESCRIP"].str.strip()
u["SIZE"] = u["SIZE"].astype(str).str.strip()

# '~' prefix in Dominick's UPC files marks discontinued items
u["discontinued"] = u["DESCRIP"].str.startswith("~").astype(int)
d = u["DESCRIP"].str.lstrip("~$").str.upper()

# ---------------------------------------------------------------- segment
SEGMENT = {
    152: "grated",
    153: "natural_chunk",
    154: "shredded_sliced",
    155: "processed_american",
    156: "specialty",
    157: "cream_cheese",
    61: "specialty",
}
u["segment"] = u["COM_CODE"].map(SEGMENT)

# ---------------------------------------------------------------- brand
# (regex on start of description, brand, firm). Order matters: first match wins.
BRAND_RULES = [
    (r"^(VELVEETA)", "Velveeta", "Kraft"),
    (r"^KR\.? ?VELVEETA|^KRAFT .*VELVEETA|^KR .*VELVE|^KRAFT .*VLVTA|^KRAFT SHRD MLD VELVE", "Velveeta", "Kraft"),
    (r"^(PHILLY|PHILADELPHIA|SOFT PHILLY)|^KR\.?.*(PHIL)", "Philadelphia", "Kraft"),
    (r"^(CRACKER BURREL)|^KR(AFT|FT)? (CR BRL|CRACKER BARREL|EX SHRP CKR|SHRP CRKR)", "Cracker Barrel", "Kraft"),
    (r"^LIGHT-N-LIVELY|^KR LIGHT & LIVELY", "Light n' Lively", "Kraft"),
    (r"^(KR|KRAFT|KRFT)\b.*HANDI|^HANDI", "Handi-Snacks", "Kraft"),
    (r"^(KR|KRAFT|KRFT)\b.*(HLTHY FAV|HEALTHY FAV|HEALHTY FAV)", "Kraft Healthy Favorites", "Kraft"),
    (r"^(KR|KRAFT|KRFT)\b", "Kraft", "Kraft"),
    (r"^ATHENOS", "Athenos", "Kraft"),
    (r"^(DOM|DOMINICK)", "Dominick's", "Dominick's (PL)"),
    (r"^HH\b", "Heritage House", "Dominick's (PL)"),
    (r"^~?GENERIC", "Generic", "Dominick's (PL)"),
    (r"^(SARGENTO|SARG|SRG)\b.*MOOTOWN|^MOOTOWN", "MooTown", "Sargento"),
    (r"^(SARGENTO|SARG|SRG)\b", "Sargento", "Sargento"),
    (r"^(CTY|CNTY|COUNTY)", "County Line", "County Line"),
    (r"^HC\b|^HEALTHY CHOICE", "Healthy Choice", "ConAgra"),
    (r"^(C ?F)\b", "CF", "Other"),
    (r"^(BORDEN|BORDON|BRDN)", "Borden", "Borden"),
    (r"^LOL\b", "Land O Lakes", "Land O Lakes"),
    (r"^(FRIGO|STELLA)", "Frigo/Stella", "Stella Foods"),
    (r"^SORRENTO", "Sorrento", "Sorrento"),
    (r"^(DORMAN|DORMN)", "Dorman", "Dorman"),
    (r"^(KAUKAUNA|KAUK)", "Kaukauna", "Kaukauna"),
    (r"^(LAUGH|LC\b)", "Laughing Cow", "Bel"),
    (r"^(WW|W\.W)\b", "Weight Watchers", "Heinz"),
    (r"^(ALPINE)", "Alpine Lace", "Alpine Lace"),
    (r"^(MILLER|MLR)", "Miller's", "Miller's"),
    (r"^MIGDAL", "Migdal", "Migdal"),
    (r"^HAOLAM", "Haolam", "Haolam"),
    (r"^FORMAGG", "Formagg", "Galaxy"),
    (r"^LEAN", "Lean n' Free", "Other"),
    (r"^(SWISS KNIGHT|SW KNIGHT)", "Swiss Knight", "Other"),
    (r"^TREASURE", "Treasure Cave", "Other"),
    (r"^RONDELE", "Rondele", "Other"),
    (r"^TRAD\b", "Traditional (cookie dough)", "Other"),
]


def brand_of(s):
    for pat, b, f in BRAND_RULES:
        if re.search(pat, s):
            return b, f
    return "Other", "Other"


bf = d.apply(brand_of)
u["brand"] = bf.str[0]
u["firm"] = bf.str[1]

# ---------------------------------------------------------------- cheese type
TYPE_RULES = [
    (r"CREAM|NEUFCH|PHIL|TEMPTEE", "cream"),
    (r"PARM|ROMAN|GRATED|GRTD", "parmesan/grated"),
    (r"STRING|STRG|STRIP|STICK", "string/stick"),
    (r"TACO|MEX|NACHO", "mexican blend"),
    (r"PIZZA|4 CH|5-CH|6 CH|3 CH|4 QT|SALAD|FOUR CHEESE|GRMT|GOURMET", "multi-cheese blend"),
    (r"MOZZ|MOZAR", "mozzarella"),
    (r"COLBY.*(JA|JK|MONT)|CHEDDAR.*MONT|CHEDDARELLA|CHEDARELLA|CHEDAR MONT|CHEDDAR & MONT|CHEDDAR/MONT", "colby-jack/cheddar-jack"),
    (r"CHED|CHD|CHDR|CHDDR|CHEDR|N\.Y|NEW YORK|SHARP|SHRP", "cheddar"),
    (r"COLBY", "colby"),
    (r"MONT|JACK|JCK", "monterey jack"),
    (r"SWISS|SWS|SWIS", "swiss"),
    (r"MUENST|MEUNST", "muenster"),
    (r"AMER|SINGLE|SNGL|DELUXE|DLX|VELVEETA|PROC|SLICE", "american/processed"),
    (r"FETA", "feta"),
    (r"GOUDA|EDAM", "gouda/edam"),
    (r"BRICK", "brick"),
    (r"PROVOLONE", "provolone"),
    (r"CHEEZ N|CRACKER|CRAC\b|CR$|HANDI|PRET", "cheese & crackers"),
    (r"COOKI|CHIP CK", "not cheese"),
    (r"SPREAD|BALLS|PORT ?WIN|WHIZ|GARDEN VEG|COL SPREAD|LIMBURGER", "spread/cold-pack"),
    (r"BRIE|CAMEMB|GOAT|HAVARTI|TILSITER|FARMER|BABYBEL|BONBEL|LAUGH|LC SHIPPER|HORSER|HORSE|HOR\b|GARLIC|SMOKED|SMKD|FONDUE|FULL MOON|CRUMBL|SQUARE|SAGO", "specialty/other natural"),
    (r"SINGL|SIN\b|INDIV|INDVL|IND\b|IND\.|WRAPP|WRPD|AM SL|LITE LINE|LITE-LINE|THICK|LOW CHLESTOL|FF SINGL|LO SOD|LOW SODIUM|SL\.?HIC|YOG", "american/processed"),
    (r"MILD|MLD|MID\b|SHA$|SHRED$|SHREDDE$|SHRED CH|SHRED.*CH$|HALFMOON|HLFMOON|NAT P|FOR$|CHEESE$|CHS$|CHSE$", "cheddar"),
    (r"\bMO\b|\bMOZ\b|SKIM|PZA", "mozzarella"),
]

# Second pass for truncated descriptions that the rules above miss
TYPE_RULES_2 = [
    (r"PEANUT BUTTER|PNT BUTTER", "not cheese"),
    (r"GRT PA|CHSE PAR|100% SHREDDED", "parmesan/grated"),
    (r"EXTRA SH|SHAR\b|NAT.*MIL$|RESEALABLE MIL|FINE SHRED MIL|SHREDDED MI$|SHRD CHE$|MEDIUM CHE", "cheddar"),
    (r"COLB$", "colby"),
    (r"\bM J$|THIN MON$", "monterey jack"),
    (r"MUEN|^DOM$", "muenster"),
    (r"SW S$", "swiss"),
    (r"EDA$|GOUD", "gouda/edam"),
    (r"TWI$", "string/stick"),
    (r"CHEESE AND|CHEESE & ST|SNACK PACK", "cheese & crackers"),
    (r"OLD ENG|LIGHT & LIVELY|VLVTA|FREE N$", "american/processed"),
    (r"SWEET KIS|FRIENDSHIP|HOT PEPPER|KNIGHT", "specialty/other natural"),
    (r"WIN SCHULER", "spread/cold-pack"),
]

# Collapse rare varieties into one bucket for a compact taxonomy
SPECIALTY = {"brick", "provolone", "feta", "gouda/edam", "spread/cold-pack",
             "specialty/other natural"}


def type_of(s, seg):
    if seg == "cream_cheese":
        return "cream"
    for pat, t in TYPE_RULES + TYPE_RULES_2:
        if re.search(pat, s):
            return t
    if seg == "processed_american":
        return "american/processed"
    return "unknown"


u["cheese_type"] = [type_of(s, g) for s, g in zip(d, u["segment"])]
u["cheese_group"] = u["cheese_type"].where(~u["cheese_type"].isin(SPECIALTY), "specialty")

# ---------------------------------------------------------------- size in oz
def size_oz(s):
    s = s.upper().replace(" ", "")
    if "TO1" in s or "#" in s or "-" in s:  # random-weight deli items / ranges
        return np.nan
    m = re.match(r"^(\d*\.?\d+)(OZ|O|LB|CT)?", s)
    if not m:
        return np.nan
    x, unit = float(m.group(1)), (m.group(2) or "OZ")
    if unit == "LB":
        return x * 16
    if unit == "CT":
        return np.nan  # count packs (snack packs): weight unknown
    return x


u["size_oz"] = u["SIZE"].apply(size_oz)
# Random-weight (store-cut, weighed, pre-priced) items: UPC number system 2,
# 11 digits, last 5 digits (price/weight field) zeroed out.
upc_s = u["UPC"].astype(str)
random_weight = (upc_s.str.len() == 11) & upc_s.str.startswith("2") & upc_s.str.endswith("00000")
u.loc[random_weight, "size_oz"] = np.nan
u["size_unit"] = np.select(
    [random_weight, u["SIZE"].str.contains("CT"), u["size_oz"].notna()],
    ["random_weight", "count", "oz"],
    "unknown",
)

# junk descriptions
u["junk_descrip"] = d.isin(["END", "2[13~"]).astype(int)

u = u.rename(columns=str.lower)
u.to_csv(OUT / "upc_clean.csv", index=False)
print(f"wrote {len(u)} UPCs to {OUT/'upc_clean.csv'}")
