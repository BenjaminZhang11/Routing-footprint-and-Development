"""Combine delegated, announced, and reachable address space into one analysis.

This is the full hierarchy. Delegated space is what a registry assigned to an
organisation in a country. Announced space is what appears in the global routing
table. Reachable space is what an active probe can actually get to. Each stage
is a strictly narrower claim than the one before it, and the drop between stages
is the quantity of interest.

Reachability is measured three ways, from loose to strict, because the loose
measure can flatter countries holding very large contiguous allocations. Any
result that survives all three is robust; any that does not should be reported
as sensitive to the definition.
"""

import ipaddress
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

PROC = Path("data/processed")
RESULTS = Path("results")

TRACES = PROC / "trace_table.csv"
PANEL = PROC / "gap_panel_ofc.csv"
FALLBACK_PANEL = PROC / "gap_panel.csv"
OUT = PROC / "full_panel.csv"


def ip_int(series):
    def conv(a):
        try:
            return int(ipaddress.IPv4Address(str(a)))
        except Exception:
            return np.nan
    return series.map(conv)


def reach_measures():
    """Compute loose, /16, and /24 reachability rates per destination country."""
    df = pd.read_csv(TRACES)
    print(f"traces loaded {len(df):,}")

    df = df[df["dst_cc"].notna()].copy()
    have_hop = df["last_hop_ip"].notna()

    df["dst_int"] = ip_int(df["dst"])
    df["hop_int"] = np.nan
    df.loc[have_hop, "hop_int"] = ip_int(df.loc[have_hop, "last_hop_ip"])

    df["reach_country"] = df["last_hop_cc"].notna() & (df["last_hop_cc"] == df["dst_cc"])
    same16 = (df["dst_int"] // 65536) == (df["hop_int"] // 65536)
    same24 = (df["dst_int"] // 256) == (df["hop_int"] // 256)
    df["reach_16"] = same16.fillna(False)
    df["reach_24"] = same24.fillna(False)

    agg = df.groupby("dst_cc").agg(
        traces=("dst", "size"),
        reach_country=("reach_country", "mean"),
        reach_16=("reach_16", "mean"),
        reach_24=("reach_24", "mean"),
        median_rtt=("rtt_ms", "median"),
    ).reset_index().rename(columns={"dst_cc": "iso2"})

    for c in ["reach_country", "reach_16", "reach_24"]:
        agg[c] = agg[c].round(4)
    agg["median_rtt"] = agg["median_rtt"].round(1)

    print(f"countries with traces {len(agg)}")
    print("\noverall reachability by definition")
    print(f"  same country {df['reach_country'].mean():.3f}")
    print(f"  same /16     {df['reach_16'].mean():.3f}")
    print(f"  same /24     {df['reach_24'].mean():.3f}\n")
    return agg


def load_panel():
    path = PANEL if PANEL.exists() else FALLBACK_PANEL
    df = pd.read_csv(path)
    print(f"panel loaded from {path.name}, {len(df)} countries")
    return df


def fit(df, formula, label, min_traces=None):
    m = smf.ols(formula, data=df).fit(cov_type="HC3")
    print("=" * 72)
    print(label)
    print("=" * 72)
    print(f"n {int(m.nobs)}   R2 {m.rsquared:.3f}")
    print(pd.DataFrame({"coef": m.params, "std_err": m.bse,
                        "p": m.pvalues}).round(4).to_string())
    print()
    return m


def main():
    agg = reach_measures()
    panel = load_panel()

    df = panel.merge(agg, on="iso2", how="left")
    matched = df["traces"].notna().sum()
    print(f"panel countries with trace data {int(matched)}\n")

    df["log_gdp_pc"] = np.log(df["gdp_per_capita_usd"])
    df["log_pop"] = np.log(df["population"])

    sub = df[(df["traces"] >= 30) & df["log_gdp_pc"].notna()].copy()
    print(f"estimation sample, at least 30 traces {len(sub)}\n")

    print("correlations among the three reachability definitions")
    print(sub[["reach_country", "reach_16", "reach_24"]].corr().round(3).to_string())
    print()

    fit(sub, "reach_country ~ log_gdp_pc + log_pop",
        "reachability, same country definition")
    fit(sub, "reach_24 ~ log_gdp_pc + log_pop",
        "reachability, strict same /24 definition")
    fit(sub, "median_rtt ~ log_gdp_pc + log_pop",
        "median round trip time to last responding hop")

    if "announcement_ratio" in sub.columns:
        print("the three stages, mean by income group")
        cols = ["announcement_ratio", "reach_country", "reach_24", "median_rtt"]
        cols = [c for c in cols if c in sub.columns]
        if "income_group" in sub.columns:
            print(sub.groupby("income_group")[cols].mean().round(3).to_string())
            print()

    show = ["iso2", "country_name", "gdp_per_capita_usd", "announcement_ratio",
            "reach_country", "reach_24", "median_rtt", "traces"]
    show = [c for c in show if c in sub.columns]

    print("lowest reachability, strict definition")
    print(sub.nsmallest(12, "reach_24")[show].round(3).to_string(index=False))

    print("\nhighest reachability, strict definition")
    print(sub.nlargest(10, "reach_24")[show].round(3).to_string(index=False))

    RESULTS.mkdir(exist_ok=True)
    df.to_csv(OUT, index=False)
    df.to_csv(RESULTS / "full_panel.csv", index=False)
    print(f"\nwritten to {OUT} and results/full_panel.csv")


if __name__ == "__main__":
    main()