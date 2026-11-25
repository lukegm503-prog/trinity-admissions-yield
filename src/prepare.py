"""Turn the raw admissions export into model-ready features.

Everything the cleaning step learns (which categories are too rare to keep, fill-in values for
missing data, clipping bounds) is learned from the training set only and then applied
unchanged to the test set. The original notebook learned these separately on each set, which
let test-set information leak into the test features and gave the two sets different
category groupings.
"""
import numpy as np
import pandas as pd

from config import RARE_CATEGORY_MIN, REFERENCE

# SAT (evidence-based reading/writing + math) to ACT composite, from the 2018 ACT/SAT
# concordance table. Each SAT cutoff maps to the ACT score for scores at or above it.
SAT_TO_ACT = {1570: 36, 1530: 35, 1490: 34, 1450: 33, 1420: 32, 1390: 31, 1360: 30, 1330: 29,
              1300: 28, 1260: 27, 1230: 26, 1200: 25, 1160: 24, 1130: 23, 1100: 22, 1060: 21,
              1030: 20, 990: 19, 960: 18, 920: 17, 880: 16, 830: 15, 780: 14, 730: 13,
              690: 12, 650: 11, 620: 10, 590: 9}

BUSINESS_MAJORS = {
    "Business", "Finance", "Business - Communication Management", "Business - Accounting",
    "Business - Sport Management", "Business - Management", "Business - Marketing",
    "Business - International Business", "Business Analytics & Technology",
    "Business - Management Information Systems", "Business Legal Studies",
}
CHRISTIAN_SMALL = {"Bible Churches", "Christian Reformed", "Christian Scientist", "Church of Christ", "Church of God"}

# Merit award codes are a letter prefix plus a dollar amount in thousands (e.g. T23 = Trustee's
# $23,000). The lookup table supplied with the data only lists 31 codes, while the data uses
# many more (P23, T22, M30, ...), so codes are parsed by prefix instead of looked up.
MERIT_PREFIXES = [("TT", "Trinity Tiger"), ("T", "Trustee's"), ("P", "President's"),
                  ("M", "Murchison"), ("D", "Dean's"), ("I", "International")]
MERIT_SPECIAL = {"SEM": "Full ride", "TTS": "Full ride", "X0": "Tuition exchange",
                 "Y0": "Tuition remission", "Z0": "No merit", "I0": "No merit"}

CATEGORICAL = ["decision_plan", "application_source", "first_source", "academic_interest_1",
               "academic_interest_2", "sport", "sport_rating", "geo_market", "citizenship",
               "merit_tier", "sex", "ethnicity", "race", "religion"]
NUMERIC = ["gpa", "act", "act_missing", "class_top_percent", "class_rank_missing",
           "academic_index", "days_first_contact_to_apply", "days_inquiry_to_apply", "no_inquiry",
           "event_count", "campus_visits", "merit_amount_k", "intends_financial_aid",
           "test_optional", "legacy", "athlete", "fine_arts", "vip"]
# "Opt Out" appears in the Legacy and Athlete fields for 750 training admits, and only 1 of them
# enrolled. The data dictionary doesn't define it, and a near-zero enrollment rate suggests it is
# recorded after a student declines. It is excluded from the model as likely target leakage;
# train.py measures how much it would inflate the results.
LEAKY = ["opt_out"]
CLIP_COLUMNS = ["days_first_contact_to_apply", "days_inquiry_to_apply"]


def parse_merit(code):
    """Return (tier, amount in $ thousands) for a merit award code."""
    if pd.isna(code):
        return "No merit", 0.0
    code = str(code).strip()
    if code in MERIT_SPECIAL:
        return MERIT_SPECIAL[code], 0.0
    for prefix, tier in MERIT_PREFIXES:
        rest = code[len(prefix):]
        if code.startswith(prefix) and rest.replace(".", "", 1).isdigit():
            amount = float(rest)
            if prefix == "TT" and amount > 100:  # "TT125" is the $12,500 Tiger award
                amount /= 10
            return tier, amount
    return "Other", 0.0


def merit_v1(code):
    """The original notebook's grouping, kept only to measure the effect of fixing it.
    Codes missing from the lookup table fell through to 'None', and duplicated keys sent
    I0 and TTS (a full ride) to 'Domestic'."""
    groups = {"International": ["I10", "I12", "I12.5", "I15", "I17", "I18", "I19", "I20", "I21", "I24",
                                "I25", "I26", "I27", "I28", "I30", "I32", "I33", "I35", "I38", "I5", "I9", "I0"],
              "Full Ride": ["SEM", "TTS"], "Full Tuition": ["X0", "Y0"],
              "Domestic": ["D12.5", "Z0", "I0", "M24", "P17", "TT9", "TTS", "T21"]}
    lookup = {c: g for g, codes in groups.items() for c in codes}
    return lookup.get(code, "None")


def _tags(row):
    tags = set()
    for col in ("Legacy", "Athlete"):
        if pd.notna(row[col]):
            tags.update(t.strip() for t in str(row[col]).split(","))
    return tags


def _sat_to_act(sat):
    if pd.isna(sat):
        return np.nan
    for cutoff, act in sorted(SAT_TO_ACT.items(), reverse=True):
        if sat >= cutoff:
            return act
    return np.nan


def base_features(df, merit="parsed"):
    """Row-by-row transformations that need no information from other rows."""
    out = pd.DataFrame(index=df.index)
    submitted = pd.to_datetime(df["Submitted"], format="mixed", errors="coerce")
    first = pd.to_datetime(df["First_Source Origin First Source Date"], format="mixed", errors="coerce")
    inquiry = pd.to_datetime(df["Inquiry Date"], format="mixed", errors="coerce")
    out["days_first_contact_to_apply"] = (submitted - first).dt.days
    out["days_inquiry_to_apply"] = (submitted - inquiry).dt.days
    out["no_inquiry"] = inquiry.isna().astype(int)

    plan = df["Decision Plan"].replace({"Early Action I": "Early Action", "Early Action II": "Early Action"})
    out["decision_plan"] = plan
    out["application_source"] = df["Application Source"]
    out["first_source"] = df["First_Source Origin First Source Summary"]

    ai1 = df["Academic Interest 1"].fillna(df["Academic Interest 2"]).fillna("Undecided")
    ai2 = df["Academic Interest 2"].where(df["Academic Interest 2"] != ai1).fillna("Undecided")
    out["academic_interest_1"] = ai1.where(~ai1.isin(BUSINESS_MAJORS), "Business")
    out["academic_interest_2"] = ai2.where(~ai2.isin(BUSINESS_MAJORS), "Business")

    tags = df.apply(_tags, axis=1)
    for name, tag in [("legacy", "Legacy"), ("athlete", "Athlete"), ("fine_arts", "Fine Arts"),
                      ("vip", "VIP"), ("opt_out", "Opt Out")]:
        out[name] = tags.apply(lambda t: int(tag in t))
    sport = df["Sport 1 Sport"].fillna("None").str.replace(" Men", "").str.replace(" Women", "")
    out["sport"] = sport
    out["sport_rating"] = df["Sport 1 Rating"].fillna("None")

    out["event_count"] = df["Total Event Participation"].clip(upper=3)
    out["campus_visits"] = df["Count of Campus Visits"].clip(upper=5)

    rank, size = df["School 1 Class Rank (Numeric)"], df["School 1 Class Size (Numeric)"]
    out["class_top_percent"] = 100 * rank / size
    out["class_rank_missing"] = out["class_top_percent"].isna().astype(int)
    out["academic_index"] = df["Academic Index"]
    out["gpa"] = df["School 1 GPA Recalculated"]

    act = df["ACT Composite"].fillna(df["SAT R Evidence-Based Reading and Writing Section + Math Section"].apply(_sat_to_act))
    out["act"] = act
    out["act_missing"] = act.isna().astype(int)
    out["test_optional"] = (df["Test Optional"] == 1).astype(int)

    regions = pd.read_csv(REFERENCE / "regions.csv").set_index("State")["Region"].str.split(",").str[0]
    state_names = {"AK": "Alaska", "AL": "Alabama", "AR": "Arkansas", "AZ": "Arizona", "CA": "California",
                   "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware", "FL": "Florida", "GA": "Georgia",
                   "HI": "Hawaii", "IA": "Iowa", "ID": "Idaho", "IL": "Illinois", "IN": "Indiana", "KS": "Kansas",
                   "KY": "Kentucky", "LA": "Louisiana", "MA": "Massachusetts", "MD": "Maryland", "ME": "Maine",
                   "MI": "Michigan", "MN": "Minnesota", "MO": "Missouri", "MS": "Mississippi", "MT": "Montana",
                   "NC": "North Carolina", "ND": "North Dakota", "NE": "Nebraska", "NH": "New Hampshire",
                   "NJ": "New Jersey", "NM": "New Mexico", "NV": "Nevada", "NY": "New York", "OH": "Ohio",
                   "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island",
                   "SC": "South Carolina", "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas", "UT": "Utah",
                   "VA": "Virginia", "VT": "Vermont", "WA": "Washington", "WI": "Wisconsin",
                   "WV": "West Virginia", "WY": "Wyoming", "DC": "D.C."}
    market = df["Permanent Geomarket"].fillna("")
    state = market.str.split("-").str[0]
    # Texas is two-thirds of applicants, so keep its individual geomarkets (distance from San
    # Antonio matters); other states roll up to their Census region.
    out["geo_market"] = np.where(
        state == "TX", "TX " + market.str.split("-").str[1].fillna(""),
        np.where(state == "INT", "International",
                 "Out of state: " + state.map(state_names).map(regions).fillna("Unknown")))
    out["citizenship"] = df["Citizenship Status"]
    out["intends_financial_aid"] = df["Intend to Apply for Financial Aid?"]

    if merit == "parsed":
        parsed = df["Merit Award"].apply(parse_merit)
        out["merit_tier"] = parsed.str[0]
        out["merit_amount_k"] = parsed.str[1]
    else:
        out["merit_tier"] = df["Merit Award"].apply(merit_v1)
        out["merit_amount_k"] = 0.0

    out["sex"] = df["Sex"]
    out["ethnicity"] = df["Ethnicity"].fillna("Not specified")
    out["race"] = df["Race"].fillna("Not specified")
    religion = df["Religion"].fillna("Not specified")
    out["religion"] = religion.where(~religion.isin(CHRISTIAN_SMALL), "Christian")
    return out


class Preprocessor:
    """Learns fill-in values, rare-category groupings, and clipping bounds from training rows."""

    def __init__(self, merit="parsed"):
        self.merit = merit

    def fit(self, raw_train):
        X = base_features(raw_train, self.merit)
        self.keep = {c: set(X[c].value_counts()[lambda s: s >= RARE_CATEGORY_MIN].index) for c in CATEGORICAL}
        self.academic_index_fill = X["academic_index"].mode()[0]
        ai = X["academic_index"].fillna(self.academic_index_fill)
        self.top_percent_by_index = X.groupby(ai)["class_top_percent"].mean()
        self.top_percent_fill = X["class_top_percent"].mean()
        self.fills = {"gpa": X["gpa"].median(), "act": X["act"].mean(),
                      "days_first_contact_to_apply": X["days_first_contact_to_apply"].median(),
                      "days_inquiry_to_apply": X["days_inquiry_to_apply"].median(),
                      "intends_financial_aid": X["intends_financial_aid"].mode()[0]}
        self.clip = {c: (X[c].quantile(0.01), X[c].quantile(0.99)) for c in CLIP_COLUMNS}
        return self

    def transform(self, raw):
        X = base_features(raw, self.merit)
        for c in CATEGORICAL:
            X[c] = X[c].where(X[c].isin(self.keep[c]), "Other")
        X["academic_index"] = X["academic_index"].fillna(self.academic_index_fill)
        by_index = X["academic_index"].map(self.top_percent_by_index).fillna(self.top_percent_fill)
        X["class_top_percent"] = X["class_top_percent"].fillna(by_index)
        for c, v in self.fills.items():
            X[c] = X[c].fillna(v)
        for c, (lo, hi) in self.clip.items():
            X[c] = X[c].clip(lo, hi)
        return X[NUMERIC + CATEGORICAL + LEAKY]


def load_split(path):
    df = pd.read_csv(path)
    train, test = df[df["train-test"] == "train"], df[df["train-test"] == "test"]
    return train, test, train["Decision"].values, test["Decision"].values
