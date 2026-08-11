"""Join RIR delegated resource counts to World Bank indicators, one row per country."""

from pathlib import Path

import numpy as np
import pandas as pd

PROC = Path("data/processed")

RIR_FILE = PROC / "rir_country.csv"
WB_FILE = PROC / "worldbank_country.csv"
OUT_FILE = PROC / "country_panel.csv"


def load():
    rir = pd.read_csv(RIR_FILE)
    wb = pd.read_csv(WB_FILE)
    rir["iso2"] = rir["iso2"].str.strip().str.upper()
    wb["iso2"] = wb["iso2"].str.strip().str.upper()
    return rir, wb


def report_unmatched(rir, wb):
    """Show which countries appear in one source but not the other."""
    only_rir = sorted(set(rir["iso2"]) - set(wb["iso2"]))
    only_wb = sorted(set(wb["iso2"]) - set(rir["iso2"]))

    print(f"in RIR only  {len(only_rir)}")
    if only_rir:
        top = rir[rir["iso2"].isin(only_rir)].nlargest(10, "ipv4_addresses")
        print(top[["iso2", "ipv4_addresses", "asn_count"]].to_string(index=False))

    print(f"\nin World Bank only  {len(only_wb)}")
    if only_wb:
        sub = wb[wb["iso2"].isin(only_wb)]
        cols = [c for c in ["iso2", "country_name", "population"] if c in sub.columns]
        print(sub[cols].to_string(index=False))


def derive(df):
    """Add per-capita and log-transformed columns used by the regression."""
    df["ipv4_per_capita"] = df["ipv4_addresses"] / df["population"]
    df["asn_per_million"] = df["asn_count"] / (df["population"] / 1e6)
    df["ipv4_blocks_per_million"] = df["ipv4_blocks"] / (df["population"] / 1e6)

    for col in [
        "ipv4_addresses",
        "asn_count",
        "ipv4_blocks",
        "gdp_usd",
        "gdp_per_capita_usd",
        "population",
    ]:
        df[f"log_{col}"] = np.log(df[col].where(df[col] > 0))

    return df


def main():
    rir, wb = load()
    print(f"RIR rows {len(rir)}   World Bank rows {len(wb)}\n")

    report_unmatched(rir, wb)

    merged = wb.merge(rir, on="iso2", how="inner")
    print(f"\nmatched countries {len(merged)}")

    merged = derive(merged)

    core = ["gdp_per_capita_usd", "population", "internet_users_pct", "ipv4_addresses"]
    complete = merged.dropna(subset=core)
    print(f"complete cases on core variables {len(complete)}")

    merged.to_csv(OUT_FILE, index=False)
    print(f"written to {OUT_FILE}\n")

    show = ["iso2", "country_name", "gdp_per_capita_usd", "population",
            "ipv4_addresses", "ipv4_per_capita", "asn_per_million"]
    print("highest IPv4 per capita")
    print(complete.nlargest(10, "ipv4_per_capita")[show].to_string(index=False))

    print("\nlowest IPv4 per capita")
    print(complete.nsmallest(10, "ipv4_per_capita")[show].to_string(index=False))

    corr = complete[["log_gdp_per_capita_usd", "log_ipv4_addresses",
                     "log_asn_count", "internet_users_pct"]].corr()
    print("\ncorrelations")
    print(corr.round(3).to_string())


if __name__ == "__main__":
    main()