"""Analyse the gap between delegated and announced IPv4 address space.

Delegated space is what a registry handed to an organisation in a country.
Announced space is what actually appears in the global routing table. The ratio
between them measures how much of a country's allocated address space is put
into use, which is the first step of the hierarchy from delegation through
allocation, routing, and reachability.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

PROC = Path("data/processed")
PANEL = PROC / "country_panel.csv"
ANNOUNCED = PROC / "announced_country.csv"
OUT = PROC / "gap_panel.csv"


def load():
    panel = pd.read_csv(PANEL)
    ann = pd.read_csv(ANNOUNCED)
    panel["iso2"] = panel["iso2"].str.strip().str.upper()
    ann["iso2"] = ann["iso2"].str.strip().str.upper()

    df = panel.merge(ann, on="iso2", how="left")
    df["announced_addresses"] = df["announced_addresses"].fillna(0)
    df["announced_prefixes"] = df["announced_prefixes"].fillna(0)

    print(f"panel countries {len(panel)}")
    print(f"with announced data {int((df['announced_addresses'] > 0).sum())}\n")
    return df


def derive(df):
    df = df.copy()
    df["announcement_ratio"] = df["announced_addresses"] / df["ipv4_addresses"]
    df["dark_addresses"] = df["ipv4_addresses"] - df["announced_addresses"]
    df["log_announced"] = np.log(df["announced_addresses"].where(
        df["announced_addresses"] > 0))
    df["announced_per_capita"] = df["announced_addresses"] / df["population"]
    return df


def describe_ratio(df):
    sub = df[(df["ipv4_addresses"] > 0) & df["announcement_ratio"].notna()]

    over = sub[sub["announcement_ratio"] > 1.05]
    print(f"countries with ratio above 1.05 {len(over)}")
    if len(over):
        cols = ["iso2", "country_name", "ipv4_addresses",
                "announced_addresses", "announcement_ratio"]
        print(over.nlargest(8, "announcement_ratio")[cols].to_string(index=False))
        print("ratios above one arise where announced prefixes extend past the")
        print("delegated block they were attributed to by starting address\n")

    valid = sub[sub["announcement_ratio"] <= 1.05]
    print(f"countries in ratio analysis {len(valid)}")
    print(valid["announcement_ratio"].describe().round(3).to_string())
    print()
    return valid


def show_extremes(valid):
    cols = ["iso2", "country_name", "gdp_per_capita_usd", "ipv4_addresses",
            "announced_addresses", "announcement_ratio"]
    big = valid[valid["ipv4_addresses"] >= 100_000]

    print("lowest share of delegated space actually announced")
    print(big.nsmallest(15, "announcement_ratio")[cols].to_string(index=False))

    print("\nhighest share of delegated space actually announced")
    print(big.nlargest(10, "announcement_ratio")[cols].to_string(index=False))
    print()


def regressions(valid):
    df = valid.dropna(subset=["log_announced", "log_gdp_per_capita_usd",
                              "log_population"])
    m = smf.ols(
        "log_announced ~ log_gdp_per_capita_usd + log_population",
        data=df).fit(cov_type="HC3")
    print("=" * 70)
    print("announced address space on income and population")
    print("=" * 70)
    print(f"n {int(m.nobs)}   R2 {m.rsquared:.3f}")
    print(pd.DataFrame({"coef": m.params, "std_err": m.bse,
                        "p": m.pvalues}).round(4).to_string())
    print()

    sub = df[(df["announcement_ratio"] > 0) & (df["announcement_ratio"] < 1)].copy()
    sub["logit_ratio"] = np.log(sub["announcement_ratio"] /
                                (1 - sub["announcement_ratio"]))
    m2 = smf.ols(
        "logit_ratio ~ log_gdp_per_capita_usd + log_population",
        data=sub).fit(cov_type="HC3")
    print("=" * 70)
    print("share of delegated space announced, on income and population")
    print("=" * 70)
    print(f"n {int(m2.nobs)}   R2 {m2.rsquared:.3f}")
    print(pd.DataFrame({"coef": m2.params, "std_err": m2.bse,
                        "p": m2.pvalues}).round(4).to_string())
    print()
    return m


def compare_residuals(valid, model):
    """Does using announced rather than delegated space change who looks unusual?"""
    df = valid.dropna(subset=["log_announced", "log_gdp_per_capita_usd",
                              "log_population"]).copy()
    df["resid_announced"] = model.resid

    if "resid_ipv4" not in df.columns:
        return df

    df["resid_shift"] = df["resid_announced"] - df["resid_ipv4"]
    cols = ["iso2", "country_name", "resid_ipv4", "resid_announced", "resid_shift"]

    print("countries that look worse once announcement is used")
    print(df.nsmallest(10, "resid_shift")[cols].round(3).to_string(index=False))

    print("\ncountries that look better once announcement is used")
    print(df.nlargest(10, "resid_shift")[cols].round(3).to_string(index=False))
    print()

    r = df[["resid_ipv4", "resid_announced"]].corr().iloc[0, 1]
    print(f"correlation between the two residual measures {r:.3f}\n")
    return df


def main():
    df = derive(load())
    valid = describe_ratio(df)
    show_extremes(valid)
    model = regressions(valid)
    out = compare_residuals(valid, model)

    keep = [c for c in [
        "iso2", "country_name", "region", "income_group",
        "gdp_per_capita_usd", "population", "internet_users_pct",
        "ipv4_addresses", "announced_addresses", "announcement_ratio",
        "dark_addresses", "announced_prefixes", "origin_asns",
        "overlap_ratio", "asn_count", "resid_ipv4", "resid_announced",
    ] if c in out.columns]
    out[keep].to_csv(OUT, index=False)
    print(f"written to {OUT}")


if __name__ == "__main__":
    main()