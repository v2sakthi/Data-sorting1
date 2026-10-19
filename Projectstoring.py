#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
generate_data.py
Creates a fake, messy lab dataset (lab_data_raw.csv) for practicing cleaning.

Scenario: a protein-concentration assay run on several instruments by
several technicians. Each row is one sample measurement.

"""

import numpy as np
import pandas as pd

rng = np.random.default_rng(seed=42)  # seed = same "random" data every time
N = 500 #How many data points can be picked


df = pd.DataFrame(
    {
        "sample_id": [f"S{i:04d}" for i in range(1, N + 1)],
        "run_date": pd.date_range("2026-01-05", periods=N, freq="6h"),  #Setting the time for eac data point
        "instrument": rng.choice(["Reader-A", "Reader-B", "Reader-C"], size=N), #Displaying which machine read the data
        "technician": rng.choice(["Alice", "Ben", "Chen", "Dana"], size=N),
        "sample_type": rng.choice(["control", "treated", "blank"], size=N, p=[0.3, 0.55, 0.15]),
        "temperature_c": rng.normal(22, 1.2, size=N).round(1),
        "ph": rng.normal(7.4, 0.15, size=N).round(2),
    }
)

# Concentration depends on sample type; absorbance depends on concentration.
base = df["sample_type"].map({"control": 2.0, "treated": 3.5, "blank": 0.1})
df["concentration_mg_ml"] = (base + rng.normal(0, 0.4, size=N)).clip(lower=0).round(3)
df["absorbance"] = (0.12 * df["concentration_mg_ml"] + rng.normal(0, 0.02, size=N)).round(4)

# A run "fails" more often when the temperature drifts or pH is off.
fail_prob = 0.05 + 0.25 * ((df["temperature_c"] > 24) | (df["ph"] < 7.15))
df["run_status"] = np.where(rng.random(N) < fail_prob, "fail", "pass")

# --- 2. Plant realistic mess -------------------------------------------------
# (a) Inconsistent text: different capitalization and stray spaces
df.loc[rng.choice(N, 40, replace=False), "instrument"] = "reader-a"
df.loc[rng.choice(N, 30, replace=False), "technician"] = " alice "
df.loc[rng.choice(N, 25, replace=False), "run_status"] = "PASS"
df.loc[rng.choice(N, 15, replace=False), "sample_type"] = "Treated"

# (b) Missing values (blank cells, plus a text placeholder)
df.loc[rng.choice(N, 35, replace=False), "concentration_mg_ml"] = np.nan
df.loc[rng.choice(N, 20, replace=False), "ph"] = np.nan
df["temperature_c"] = df["temperature_c"].astype(object)
df.loc[rng.choice(N, 12, replace=False), "temperature_c"] = "N/A"

# (c) Outliers: sensor glitches
df.loc[rng.choice(N, 5, replace=False), "absorbance"] = 9.99
df.loc[rng.choice(N, 3, replace=False), "concentration_mg_ml"] = -1.0

# (d) Dates in mixed formats
df["run_date"] = df["run_date"].dt.strftime("%Y-%m-%d %H:%M")
mixed = rng.choice(N, 30, replace=False)
df.loc[mixed, "run_date"] = pd.to_datetime(df.loc[mixed, "run_date"]).dt.strftime("%d/%m/%Y")

# (e) Duplicate rows (someone exported twice)
df = pd.concat([df, df.sample(10, random_state=1)], ignore_index=True)

# --- 3. Save -----------------------------------------------------------------
df = df.sample(frac=1, random_state=7).reset_index(drop=True)  # shuffle rows
df.to_csv("lab_data_raw.csv", index=False)
print(f"Wrote lab_data_raw.csv with {len(df)} rows and {df.shape[1]} columns")


# In[2]:


"""
inspect_data.py
Stage 2 of the project: load the raw lab data and look at it BEFORE cleaning.
This is to ensure that all data is properly labelled and so I can see the resulting data in it's own columns

"""

import pandas as pd

# Showing every column when printing (otherwise pandas hides the middle with "...").
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)

# A DataFrame is pandas' table: rows are samples, columns are measurements.
df = pd.read_csv("lab_data_raw.csv")


def section(title):
    """Print a header so the output is easy to scan."""
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")



section("1. Shape (rows, columns)")
print(df.shape)

section("2. First 5 rows")
print(df.head())  # use df.tail() for the last rows, df.sample(5) for random ones


section("3. Column data types")
# 'object' means text. If a column that SHOULD be numeric shows 'object',
# something non-numeric (like "N/A") is hiding in it.
print(df.dtypes)


section("4. Missing values per column")
# df.isna() gives True/False for every cell; .sum() counts the Trues per column.
print(df.isna().sum())


section("5. Duplicate rows")
print("Fully duplicated rows:", df.duplicated().sum())


section("6. Unique values in text columns")
# value_counts() counts how often each label appears.
for col in ["instrument", "technician", "sample_type", "run_status"]:
    print(f"\n{col}:")
    print(df[col].value_counts(dropna=False))


section("7. Summary statistics for numeric columns")
print(df.describe())


section("8. Text placeholders hiding in temperature_c")
raw = pd.read_csv("lab_data_raw.csv", keep_default_na=False)
# pd.to_numeric(..., errors="coerce") turns anything unparseable into NaN.

as_numbers = pd.to_numeric(raw["temperature_c"], errors="coerce")
bad_values = raw.loc[as_numbers.isna(), "temperature_c"]
print(bad_values.value_counts())
print("\nLesson: always check what the raw file really contains; pandas may")
print("have already 'cleaned' something for you without telling you.")


section("9. Date formats")
looks_iso = df["run_date"].str.match(r"^\d{4}-\d{2}-\d{2}")
print("ISO style (2026-01-05 ...):", looks_iso.sum())
print("Other style (05/01/2026):  ", (~looks_iso).sum())


# In[3]:


"""

Stage 3 of the project: clean the raw lab data that was generated.

"""

import numpy as np
import pandas as pd

RAW_FILE = "lab_data_raw.csv"
CLEAN_FILE = "lab_data_clean.csv"

# An absorbance above this is physically implausible, so I'm
# treating it as a sensor glitch. 
MAX_ABSORBANCE = 3.0



def remove_duplicates(df):
    before = len(df)
    # drop_duplicates() keeps the first copy of any row that is identical
    # to an earlier row in EVERY column.
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Duplicate rows removed: {before - len(df)}")
    return df

def standardize_text(df):
    df = df.copy()  # work on a copy so the input DataFrame is left untouched

    for col in ["instrument", "technician", "sample_type", "run_status"]:
        df[col] = df[col].str.strip()

    df["instrument"] = df["instrument"].str.title()
    df["technician"] = df["technician"].str.title()

    df["sample_type"] = df["sample_type"].str.lower()
    df["run_status"] = df["run_status"].str.lower()

    print("Text labels standardized.")
    return df


def parse_dates(df):
    df = df.copy()

    # errors="coerce" turns rows that don't match into NaT ("not a time").
    iso_style = pd.to_datetime(df["run_date"], format="%Y-%m-%d %H:%M", errors="coerce")
    day_first = pd.to_datetime(df["run_date"], format="%d/%m/%Y", errors="coerce")

    # fillna() fills the gaps in the first result using the second.
    #This is due to the dates bein ambiguos
    df["run_date"] = iso_style.fillna(day_first)

    unparsed = df["run_date"].isna().sum()
    print(f"Dates parsed. Rows that matched neither format: {unparsed}")
    return df


def convert_numbers(df):
    df = df.copy()
    number_cols = ["temperature_c", "ph", "concentration_mg_ml", "absorbance"]
    for col in number_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    print("Measurement columns converted to numbers.")
    return df


def remove_impossible_values(df):
    df = df.copy()

    # A comparison like (column < 0) gives a True/False value for every row.
    negative_conc = df["concentration_mg_ml"] < 0
    too_high_abs = df["absorbance"] > MAX_ABSORBANCE

    print(f"Negative concentrations set to missing: {negative_conc.sum()}")
    print(f"Absorbance above {MAX_ABSORBANCE} set to missing: {too_high_abs.sum()}")

    df.loc[negative_conc, "concentration_mg_ml"] = np.nan
    df.loc[too_high_abs, "absorbance"] = np.nan
    return df



def handle_missing(df):
    df = df.copy()

    # Rule A: concentration and absorbance ARE the experiment. A row without
    # them has nothing to analyze, so it's dropped
    before = len(df)
    df = df.dropna(subset=["concentration_mg_ml", "absorbance"])
    print(f"Rows dropped for missing concentration/absorbance: {before - len(df)}")

    # Rule B: temperature and pH are background conditions. Losing a whole
    # row for one is wasteful, so gap is filled with the column's median
    for col in ["temperature_c", "ph"]:
        n_missing = df[col].isna().sum()
        df[col] = df[col].fillna(df[col].median())
        print(f"{col}: filled {n_missing} missing values with the median")

    return df.reset_index(drop=True)


def check_result(df):
    """assert stops the script with a message if something is still wrong."""
    assert df.duplicated().sum() == 0, "duplicates remain"
    assert df["sample_id"].is_unique, "sample_id values repeat"
    assert df.isna().sum().sum() == 0, "missing values remain"
    assert (df["concentration_mg_ml"] >= 0).all(), "negative concentration remains"
    assert (df["absorbance"] <= MAX_ABSORBANCE).all(), "absorbance outlier remains"
    assert set(df["instrument"]) == {"Reader-A", "Reader-B", "Reader-C"}
    assert set(df["run_status"]) == {"pass", "fail"}
    assert set(df["sample_type"]) == {"control", "treated", "blank"}
    print("All sanity checks passed.")


def main():
    df = pd.read_csv(RAW_FILE)
    print(f"Loaded {len(df)} rows from {RAW_FILE}\n")

    df = remove_duplicates(df)
    df = standardize_text(df)
    df = parse_dates(df)
    df = convert_numbers(df)
    df = remove_impossible_values(df)
    df = handle_missing(df)

    print()
    check_result(df)

    df.to_csv(CLEAN_FILE, index=False)
    print(f"\nSaved {len(df)} clean rows to {CLEAN_FILE}")
    print(df.dtypes)



if __name__ == "__main__":
    main()


# In[4]:


import pandas as pd

df = pd.read_csv("lab_data_clean.csv")
print(df)

