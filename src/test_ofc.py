"""Test whether the announcement ratio is non-linear in income and whether
offshore financial centers behave differently from other countries.

The offshore classification is taken from Zoromé (2007), IMF Working Paper
07/87, which derives OFC status empirically from net exports of financial
services rather than by reputation. Using an externally published list avoids
selecting jurisdictions because they happen to fit the pattern in this data.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

PROC = Path("data/processed")
GAP = PROC / "gap_panel.csv"
OUT = PROC / "gap_panel_ofc.csv"

# Zoromé (2007), IMF WP/07/87, Table 7. High income group.
OFC_HIGH = {
    "BS",  # Bahamas
    "BH",  # Bahrain
    "BM",  # Bermuda
    "KY",  # Cayman Islands
    "CY",  # Cyprus
    "HK",  # Hong Kong
    "GG",  # Guernsey
    "IE",  # Ireland
    "IM",  # Isle of Man
    "JE",  # Jersey
    "LU",  # Luxembourg
    "MT",  # Malta
    "SG",  # Singapore
    "CH",  # Switzerland
    "GB",  # United Kingdom
}

# Netherlands Antilles was on the list but dissolved in 2010. Its successor
# territories are included so the classification remains applicable to
# present-day country codes.
OFC_SUCCESSORS = {"CW", "SX", "BQ", "AN"}

# Zoromé (2007), low and middle income group.
OFC_LOWMID = {
    "BB",  # Barbados
    "LV",  # Latvia
    "MU",  # Mauritius
    "PA",  # Panama
    "UY",  # Uruguay
    "VU",  # Vanuatu
}

OFC = OFC_HIGH | OFC_LOWMID | OFC_SUCCESSORS


def load():
    df = pd.read_csv(GAP)
    df["iso2"] = df["iso2"].str.strip().str.upper()
    df["ofc"] = df["iso2"].isin(OFC).astype(int)

    present = sorted(set(df.loc[df["ofc"] == 1, "iso2"]))
    missing = sorted(OFC - set(df["iso2"]))
    print(f"offshore centres in sample {len(present)}")
    print(f"codes {present}")
    print(f"on list but absent from sample {missing}\n")
    return df


def prep(df):
    df = df.copy()
    df["log_gdp_pc"] = np.log(df["gdp_per_capita_usd"])
    df["log_pop"] = np.log(df["population"])
    df["log_gdp_pc_sq"] = df["log_gdp_pc"] ** 2

    df = df[(df["announcement_ratio"] > 0) & (df["announcement_ratio"] < 1)]
    df = df.dropna(subset=["log_gdp_pc", "log_pop", "announcement_ratio"])
    df["logit_ratio"] = np.log(
        df["announcement_ratio"] / (1 - df["announcement_ratio"]))
    return df


def fit(df, formula, label):
    m = smf.ols(formula, data=df).fit(cov_type="HC3")
    print("=" * 72)
    print(label)
    print("=" * 72)
    print(f"n {int(m.nobs)}   R2 {m.rsquared:.3f}   adj R2 {m.rsquared_adj:.3f}")
    print(pd.DataFrame({"coef": m.params, "std_err": m.bse,
                        "p": m.pvalues}).round(4).to_string())
    print()
    return m


def turning_point(model):
    """Income level at which the fitted curve reverses direction."""
    if "log_gdp_pc_sq" not in model.params.index:
        return
    b1 = model.params["log_gdp_pc"]
    b2 = model.params["log_gdp_pc_sq"]
    if abs(b2) < 1e-12:
        return
    x_star = -b1 / (2 * b2)
    shape = "minimum" if b2 > 0 else "maximum"
    print(f"curve has a {shape} at log income {x_star:.2f}, "
          f"about {np.exp(x_star):,.0f} dollars per capita\n")


def group_means(df):
    print("mean announcement ratio by offshore status")
    print(df.groupby("ofc")["announcement_ratio"].agg(
        ["mean", "median", "count"]).round(3).to_string())
    print()

    if "income_group" in df.columns:
        print("mean announcement ratio by income group and offshore status")
        tab = df.groupby(["income_group", "ofc"])["announcement_ratio"].agg(
            ["mean", "count"]).round(3)
        print(tab.to_string())
        print()


def main():
    raw = load()
    df = prep(raw)
    print(f"countries in estimation sample {len(df)}\n")

    group_means(df)

    fit(df, "logit_ratio ~ log_gdp_pc + log_pop",
        "Model A  linear in income")

    mb = fit(df, "logit_ratio ~ log_gdp_pc + log_gdp_pc_sq + log_pop",
             "Model B  quadratic in income")
    turning_point(mb)

    fit(df, "logit_ratio ~ log_gdp_pc + log_pop + ofc",
        "Model C  linear in income plus offshore indicator")

    md = fit(df, "logit_ratio ~ log_gdp_pc + log_gdp_pc_sq + log_pop + ofc",
             "Model D  quadratic in income plus offshore indicator")
    turning_point(md)

    df = df.copy()
    df["resid_ratio"] = md.resid
    cols = ["iso2", "country_name", "gdp_per_capita_usd",
            "announcement_ratio", "ofc", "resid_ratio"]
    cols = [c for c in cols if c in df.columns]

    print("still unexplained after income, population, and offshore status")
    print(df.nsmallest(10, "resid_ratio")[cols].round(3).to_string(index=False))
    print()

    df.to_csv(OUT, index=False)
    print(f"written to {OUT}")


if __name__ == "__main__":
    main()