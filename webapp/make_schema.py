"""Form schema for the web pages: readable labels for every coded column (UCI dataset documentation),
the codes that appear in the data, the 10 economic cohorts, preset profiles and feature means."""
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE) + "/"
raw = pd.read_csv(ROOT + "data/raw/students_dropout_academic_success.csv")
tr = json.load(open(HERE + "/data/train_raw.json"))
train = pd.DataFrame(tr["rows"], columns=tr["columns"]); train["target"] = tr["y"]
prepared = pd.read_csv(ROOT + "data/processed/train.csv")

MARITAL = {1: "Single", 2: "Married", 3: "Widower", 4: "Divorced", 5: "Facto union", 6: "Legally separated"}
APP_MODE = {1: "1st phase – general contingent", 2: "Ordinance No. 612/93", 5: "1st phase – special contingent (Azores)",
    7: "Holders of other higher courses", 10: "Ordinance No. 854-B/99", 15: "International student (bachelor)",
    16: "1st phase – special contingent (Madeira)", 17: "2nd phase – general contingent", 18: "3rd phase – general contingent",
    26: "Ordinance No. 533-A/99, b2 (different plan)", 27: "Ordinance No. 533-A/99, b3 (other institution)",
    39: "Over 23 years old", 42: "Transfer", 43: "Change of course", 44: "Technological specialization diploma holders",
    51: "Change of institution/course", 53: "Short cycle diploma holders", 57: "Change of institution/course (international)"}
COURSE = {33: "Biofuel Production Technologies", 171: "Animation and Multimedia Design", 8014: "Social Service (evening)",
    9003: "Agronomy", 9070: "Communication Design", 9085: "Veterinary Nursing", 9119: "Informatics Engineering",
    9130: "Equinculture", 9147: "Management", 9238: "Social Service", 9254: "Tourism", 9500: "Nursing",
    9556: "Oral Hygiene", 9670: "Advertising and Marketing Management", 9773: "Journalism and Communication",
    9853: "Basic Education", 9991: "Management (evening)"}
PREV_QUAL = {1: "Secondary education", 2: "Higher education – bachelor's degree", 3: "Higher education – degree",
    4: "Higher education – master's", 5: "Higher education – doctorate", 6: "Frequency of higher education",
    9: "12th year – not completed", 10: "11th year – not completed", 12: "Other – 11th year", 14: "10th year",
    15: "10th year – not completed", 19: "Basic education 3rd cycle (9th–11th year)",
    38: "Basic education 2nd cycle (6th–8th year)", 39: "Technological specialization course",
    40: "Higher education – degree (1st cycle)", 42: "Professional higher technical course",
    43: "Higher education – master (2nd cycle)"}
NATION = {1: "Portuguese", 2: "German", 6: "Spanish", 11: "Italian", 13: "Dutch", 14: "English", 17: "Lithuanian",
    21: "Angolan", 22: "Cape Verdean", 24: "Guinean", 25: "Mozambican", 26: "Santomean", 32: "Turkish",
    41: "Brazilian", 62: "Romanian", 100: "Moldovan", 101: "Mexican", 103: "Ukrainian", 105: "Russian",
    108: "Cuban", 109: "Colombian"}
PARENT_QUAL = {1: "Secondary education (12th year)", 2: "Higher education – bachelor's degree", 3: "Higher education – degree",
    4: "Higher education – master's", 5: "Higher education – doctorate", 6: "Frequency of higher education",
    9: "12th year – not completed", 10: "11th year – not completed", 11: "7th year (old)", 12: "Other – 11th year",
    13: "2nd year complementary high school", 14: "10th year", 18: "General commerce course",
    19: "Basic education 3rd cycle (9th–11th year)", 20: "Complementary high school course",
    22: "Technical-professional course", 25: "Complementary high school – not concluded", 26: "7th year",
    27: "2nd cycle general high school", 29: "9th year – not completed", 30: "8th year",
    31: "General course of administration and commerce", 33: "Supplementary accounting and administration",
    34: "Unknown", 35: "Can't read or write", 36: "Can read without a 4th year", 37: "Basic education 1st cycle (4th/5th year)",
    38: "Basic education 2nd cycle (6th–8th year)", 39: "Technological specialization course",
    40: "Higher education – degree (1st cycle)", 41: "Specialized higher studies course",
    42: "Professional higher technical course", 43: "Higher education – master (2nd cycle)",
    44: "Higher education – doctorate (3rd cycle)"}
OCCUPATION = {0: "Student", 1: "Legislators, directors and executive managers", 2: "Intellectual and scientific specialists",
    3: "Intermediate-level technicians", 4: "Administrative staff", 5: "Personal services, security and sellers",
    6: "Farmers and skilled agriculture/fishery workers", 7: "Skilled industry and construction workers",
    8: "Machine operators and assembly workers", 9: "Unskilled workers", 10: "Armed forces", 90: "Other situation",
    99: "(blank)", 101: "Armed forces officers", 102: "Armed forces sergeants", 103: "Other armed forces personnel",
    112: "Directors of administrative and commercial services", 114: "Hotel, catering and trade directors",
    121: "Physical sciences, maths and engineering specialists", 122: "Health professionals", 123: "Teachers",
    124: "Finance, accounting and administration specialists", 125: "ICT specialists",
    131: "Science and engineering technicians", 132: "Intermediate-level health technicians",
    134: "Legal, social, sports and cultural technicians", 135: "ICT technicians",
    141: "Office workers and secretaries", 143: "Data, accounting and financial operators",
    144: "Other administrative support staff", 151: "Personal service workers", 152: "Sellers",
    153: "Personal care workers", 154: "Protection and security services", 161: "Market-oriented farmers",
    163: "Subsistence farmers and fishermen", 171: "Skilled construction workers", 172: "Metal and metalworking workers",
    173: "Printing, precision and craft workers", 174: "Electricity and electronics workers",
    175: "Food, wood and clothing industry workers", 181: "Fixed plant and machine operators", 182: "Assembly workers",
    183: "Vehicle drivers", 191: "Cleaning workers", 192: "Unskilled agriculture and fishery workers",
    193: "Unskilled industry, construction and transport workers", 194: "Meal preparation assistants",
    195: "Street vendors"}
LABELS = {"Marital Status": MARITAL, "Application mode": APP_MODE, "Course": COURSE,
          "Previous qualification": PREV_QUAL, "Nacionality": NATION, "Mother's qualification": PARENT_QUAL,
          "Father's qualification": PARENT_QUAL, "Mother's occupation": OCCUPATION, "Father's occupation": OCCUPATION}


def options(col):
    counts = raw[col].value_counts()
    missing = [int(c) for c in counts.index if int(c) not in LABELS[col]]
    assert not missing, (col, missing)
    codes = [int(c) for c in counts.index]                       # most common first
    if col in ("Course", "Marital Status"):
        codes = sorted(codes, key=lambda c: LABELS[col][c] if col == "Course" else c)
    return [[c, LABELS[col][c]] for c in codes]


eco = train.groupby(["Unemployment rate", "Inflation rate", "GDP"]).size().reset_index(name="n")
eco_list = [[float(r["Unemployment rate"]), float(r["Inflation rate"]), float(r["GDP"])] for _, r in eco.iterrows()]


def profile(df):
    p = {}
    for c in tr["columns"]:
        if c in LABELS or df[c].nunique() <= 2 or c == "Application order":
            p[c] = float(df[c].mode()[0])
        else:
            p[c] = float(round(df[c].median(), 2))
    for s in ("1st", "2nd"):
        e, a, g = (f"Curricular units {s} sem ({k})" for k in ("enrolled", "approved", "grade"))
        p[a] = min(p[a], p[e])
        if p[a] == 0:
            p[g] = 0.0
        elif p[g] == 0:
            p[g] = float(round(df.loc[df[a] > 0, g].median(), 2))
        ev = f"Curricular units {s} sem (evaluations)"
        p[ev] = max(p[ev], 0.0)
    combo = df.groupby(["Unemployment rate", "Inflation rate", "GDP"]).size().idxmax()
    p["Unemployment rate"], p["Inflation rate"], p["GDP"] = (float(v) for v in combo)
    return p


presets = {"average": profile(train)}
for cls in ("Graduate", "Dropout", "Enrolled"):
    presets[cls.lower()] = profile(train[train["target"] == cls])

schema = {
    "options": {c: options(c) for c in LABELS},
    "economy": eco_list,
    "presets": presets,
    "feature_means": prepared.drop(columns="target").mean().tolist(),
}
json.dump(schema, open(HERE + "/data/schema.json", "w"))
for k, v in presets.items():
    print(k, {c: v[c] for c in ["Curricular units 1st sem (enrolled)", "Curricular units 1st sem (approved)",
                                "Curricular units 1st sem (grade)", "Curricular units 2nd sem (approved)",
                                "Curricular units 2nd sem (grade)", "Tuition fees up to date", "Age at enrollment",
                                "Course", "Unemployment rate"]})
print({c: len(o) for c, o in schema["options"].items()})
