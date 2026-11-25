"""Create data/synthetic/tu_synthetic.csv: a stand-in for the confidential TU.csv.

It has the same 69 columns so the pipeline runs end to end, but no real applicant's record:
  * Columns are sampled independently in small groups (test scores, high school record,
    athletics, geography, and so on), so no synthetic row matches a real person.
  * Categories with fewer than 20 real applicants are replaced with the most common value.
  * Postal codes, staff names, and dates are generated, not sampled.
  * Enrollment (Decision) is simulated from a simple made-up formula, so models trained on this
    file will not reproduce the real results in the README.

Requires the real TU.csv at data/raw/TU.csv. Run once by the data owner:
    python src/make_synthetic.py
"""
import numpy as np
import pandas as pd

from config import REAL_DATA, SYNTHETIC_DATA

N_ROWS, N_TRAIN, MIN_CATEGORY = 8000, 5300, 20
rng = np.random.default_rng(42)

BLOCKS = [
    ["Sex"], ["Ethnicity"], ["Race"], ["Religion"], ["Application Source"], ["Decision Plan"],
    ["Legacy"], ["Academic Interest 1"], ["Academic Interest 2"],
    ["First_Source Origin First Source Summary"], ["Total Event Participation"],
    ["Count of Campus Visits"], ["Academic Index"], ["Intend to Apply for Financial Aid?"],
    ["Merit Award"], ["Entry Term (Application)"],
    ["Athlete", "Sport 1 Sport", "Sport 1 Rating", "Sport 2 Sport", "Sport 2 Rating", "Sport 3 Sport", "Sport 3 Rating"],
    ["Permanent Country", "Permanent Geomarket", "Citizenship Status"],
    ["School #1 Organization Category", "School 1 Code", "School 1 Class Rank (Numeric)",
     "School 1 Class Size (Numeric)", "School 1 GPA", "School 1 GPA Scale", "School 1 GPA Recalculated",
     "School 2 Class Rank (Numeric)", "School 2 Class Size (Numeric)", "School 2 GPA", "School 2 GPA Scale",
     "School 2 GPA Recalculated", "School 3 Class Rank (Numeric)", "School 3 Class Size (Numeric)",
     "School 3 GPA", "School 3 GPA Scale", "School 3 GPA Recalculated"],
    ["ACT Composite", "ACT English", "ACT Reading", "ACT Math", "ACT Science Reasoning", "ACT Writing",
     "SAT I CR + M", "SAT R Evidence-Based Reading and Writing Section + Math Section",
     "SAT Concordance Score (of SAT R)", "ACT Concordance Score (of SAT R)", "ACT Concordance Score (of SAT)",
     "Test Optional", "SAT I Critical Reading", "SAT I Math", "SAT I Writing",
     "SAT R Evidence-Based Reading and Writing Section", "SAT R Math Section"],
]


def sample_block(real, cols):
    block = real[cols].copy()
    for c in cols:
        if block[c].dtype == object:
            counts = block[c].value_counts()
            rare = counts[counts < MIN_CATEGORY].index
            block.loc[block[c].isin(rare), c] = counts.index[0]
    if "School 1 Code" in cols:
        block["School 1 Code"] = np.nan  # school identifiers are never copied
    return block.iloc[rng.integers(0, len(block), N_ROWS)].reset_index(drop=True)


def main():
    real = pd.read_csv(REAL_DATA)
    syn = pd.concat([sample_block(real, cols) for cols in BLOCKS], axis=1)

    syn["ID"] = np.arange(1, N_ROWS + 1)
    syn["train-test"] = np.where(rng.permutation(N_ROWS) < N_TRAIN, "train", "test")
    syn["Admit Type"] = "FY"
    syn["Staff Assigned Name"] = rng.choice([f"Counselor {c}" for c in "ABCDEFGH"], N_ROWS)
    syn["Permanent Postal"] = [f"{z:05d}-0000" for z in rng.integers(10000, 99999, N_ROWS)]

    year = syn["Entry Term (Application)"].str[-4:].astype(int)
    submitted = pd.to_datetime((year - 1).astype(str) + "-10-01") + pd.to_timedelta(rng.integers(0, 140, N_ROWS), unit="D")
    first = submitted - pd.to_timedelta(rng.gamma(2.0, 150, N_ROWS).astype(int), unit="D")
    inquiry = submitted - pd.to_timedelta(rng.integers(-30, 400, N_ROWS), unit="D")
    has_inquiry = rng.random(N_ROWS) < 0.69
    syn["Submitted"] = submitted.dt.strftime("%-m/%-d/%Y")
    syn["First_Source Origin First Source Date"] = first.dt.strftime("%-m/%-d/%Y %-H:%M")
    syn["Inquiry Date"] = np.where(has_inquiry, inquiry.dt.strftime("%-m/%-d/%Y %-H:%M"), None)

    # Simulated enrollment decision (made-up coefficients, calibrated to about 22%)
    early_decision = syn["Decision Plan"].str.startswith("Early Decision").astype(int)
    merit_k = pd.to_numeric(syn["Merit Award"].str.extract(r"(\d+\.?\d*)")[0], errors="coerce").fillna(0)
    logit = (-1.7 + 1.1 * syn["Total Event Participation"].clip(upper=3) + 0.6 * syn["Count of Campus Visits"].clip(upper=4)
             + 1.5 * early_decision + 0.5 * syn["Athlete"].notna() - 0.04 * merit_k
             + 0.4 * syn["Permanent Geomarket"].str.startswith("TX", na=False) + rng.normal(0, 0.8, N_ROWS))
    syn["Decision"] = (rng.random(N_ROWS) < 1 / (1 + np.exp(-logit))).astype(int)

    syn = syn[real.columns]
    SYNTHETIC_DATA.parent.mkdir(parents=True, exist_ok=True)
    syn.to_csv(SYNTHETIC_DATA, index=False)
    print(f"Wrote {len(syn)} synthetic rows; enrollment rate {syn['Decision'].mean():.1%}")


if __name__ == "__main__":
    main()
