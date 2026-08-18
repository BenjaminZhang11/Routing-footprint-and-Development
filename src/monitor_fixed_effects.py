"""Estimate country-level latency and reachability with monitor fixed effects.

In Ark team probing the team divides destination space among monitors, so each
destination is probed from one vantage point rather than all of them. Raw
latency therefore confounds a destination country's infrastructure with the
accident of which monitor drew that target. Including a dummy for every monitor
absorbs the vantage point, so the remaining variation across destination
countries is not driven by measurement geography.

Latency is modelled at the trace level with monitor fixed effects, then country
effects are recovered and regressed on income. Reachability is handled the same
way. The EU code is dropped because RIPE uses it for pan-European resources
rather than for a country.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

PROC = Path("data/processed")
RESULTS = Path("results")

TRACES = PROC / "trace_table.csv"
PANEL = PROC / "gap_panel_ofc.csv"
FALLBACK = PROC / "gap_panel.csv"
OUT = PROC / "country_effects.csv"

MIN_TRACES = 30
NON_COUNTRIES = {"EU", "AP", "ZZ"}


def load_traces():
    df = pd.read_csv(TRACES)
    df = df[df["dst_cc"].notna() & ~df["dst_cc"].isin(NON_COUNTRIES)]
    df = df[df["rtt_ms"].notna() & (df["rtt_ms"] > 0)]

    # Drop implausible latencies. Anything beyond two seconds reflects a stalled
    # probe or a queueing pathology rather than path length.
    before = len(df)
    df = df[df["rtt_ms"] < 2000]
    print(f"traces {before:,}, kept after latency filter {len(df):,}")

    counts = df["dst_cc"].value_counts()
    keep = counts[counts >= MIN_TRACES].index
    df = df[df["dst_cc"].isin(keep)]
    print(f"countries with at least {MIN_TRACES} traces {len(keep)}")
    print(f"monitors {df['monitor'].nunique()}\n")

    df["log_rtt"] = np.log(df["rtt_ms"])
    df["reached"] = (df["last_hop_cc"] == df["dst_cc"]).astype(int)
    return df


def country_effects(df, outcome, label):
    """Fit outcome on country and monitor dummies, return country coefficients."""
    model = smf.ols(f"{outcome} ~ C(dst_cc) + C(monitor)", data=df).fit()
    print(f"{label}")
    print(f"  n {int(model.nobs):,}   R2 {model.rsquared:.3f}")

    params = model.params
    rows = []
    for name, value in params.items():
        if name.startswith("C(dst_cc)[T."):
            cc = name.split("T.")[1].rstrip("]")
            rows.append({"iso2": cc, label: value})
    out = pd.DataFrame(rows)

    # The omitted category sits at zero by construction. Centre the effects so
    # they are read relative to the average country rather than to whichever
    # code happened to sort first.
    out[label] = out[label] - out[label].mean()
    print(f"  countries recovered {len(out)}\n")
    return out


def load_panel():
    path = PANEL if PANEL.exists() else FALLBACK
    df = pd.read_csv(path)
    print(f"panel {path.name}, {len(df)} countries\n")
    return df


def fit(df, formula, label):
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
    df = load_traces()

    rtt_fx = country_effects(df, "log_rtt", "rtt_effect")
    reach_fx = country_effects(df, "reached", "reach_effect")

    raw = df.groupby("dst_cc").agg(
        traces=("dst", "size"),
        raw_median_rtt=("rtt_ms", "median"),
        raw_reach=("reached", "mean"),
    ).reset_index().rename(columns={"dst_cc": "iso2"})

    eff = raw.merge(rtt_fx, on="iso2", how="left").merge(reach_fx, on="iso2", how="left")
    panel = load_panel()
    m = panel.merge(eff, on="iso2", how="inner")
    print(f"matched to panel {len(m)}\n")

    m["log_gdp_pc"] = np.log(m["gdp_per_capita_usd"])
    m["log_pop"] = np.log(m["population"])
    sub = m.dropna(subset=["log_gdp_pc", "log_pop", "rtt_effect"]).copy()
    print(f"estimation sample {len(sub)}\n")

    print("how much the vantage point mattered")
    r = np.corrcoef(np.log(sub["raw_median_rtt"]), sub["rtt_effect"])[0, 1]
    print(f"  correlation of raw log latency with adjusted effect {r:.3f}\n")

    fit(sub, "rtt_effect ~ log_gdp_pc + log_pop",
        "adjusted latency on income and population")
    fit(sub, "reach_effect ~ log_gdp_pc + log_pop",
        "adjusted reachability on income and population")

    if "announcement_ratio" in sub.columns:
        fit(sub, "rtt_effect ~ log_gdp_pc + log_pop + announcement_ratio",
            "adjusted latency, controlling for announcement ratio")

    show = ["iso2", "country_name", "gdp_per_capita_usd", "raw_median_rtt",
            "rtt_effect", "reach_effect", "traces"]
    show = [c for c in show if c in sub.columns]

    print("lowest adjusted latency, best connected relative to vantage points")
    print(sub.nsmallest(12, "rtt_effect")[show].round(3).to_string(index=False))

    print("\nhighest adjusted latency")
    print(sub.nlargest(12, "rtt_effect")[show].round(3).to_string(index=False))

    if "income_group" in sub.columns:
        print("\nadjusted measures by income group")
        cols = [c for c in ["rtt_effect", "reach_effect", "announcement_ratio"]
                if c in sub.columns]
        print(sub.groupby("income_group")[cols].mean().round(3).to_string())

    RESULTS.mkdir(exist_ok=True)
    sub.to_csv(OUT, index=False)
    sub.to_csv(RESULTS / "country_effects.csv", index=False)
    print(f"\nwritten to {OUT} and results/country_effects.csv")


if __name__ == "__main__":
    main()