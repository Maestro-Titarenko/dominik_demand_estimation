"""Coarse product groupings shared by 02_clean.py and 03_blp_nevo.py.

Keeps the cardinality of each characteristic at or below 10: 8 brand groups, 9 type groups.
"""

BRAND_GROUP = {
    "Dominick's": "private label", "Heritage House": "private label", "Generic": "private label",
    "Kraft": "Kraft", "Cracker Barrel": "Kraft", "Kraft Healthy Favorites": "Kraft",
    "Light n' Lively": "Kraft",
    "Philadelphia": "Philadelphia", "Velveeta": "Velveeta",
    "Sargento": "Sargento", "MooTown": "Sargento",
    "County Line": "County Line", "Borden": "Borden",
}
OTHER_BRAND = "other national"


def brand_group(brand):
    return BRAND_GROUP.get(brand, OTHER_BRAND)


def type_group(cheese_group, form):
    if cheese_group == "cream":
        return "cream cheese"
    if cheese_group in ("string/stick", "cheese & crackers"):
        return "snack (string, crackers)"
    if cheese_group == "american/processed" or form == "processed":
        return "processed"
    if cheese_group == "parmesan/grated":
        return "grated parmesan"
    if form == "shredded":
        return "shredded"
    if cheese_group == "specialty" or form == "specialty":
        return "specialty"
    if cheese_group == "cheddar":
        return "cheddar block/sliced"
    if cheese_group == "mozzarella":
        return "mozzarella block/sliced"
    return "other natural block/sliced"
