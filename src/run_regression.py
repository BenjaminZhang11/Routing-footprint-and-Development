"""Log-log regressions of routing footprint on economic development."""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

PROC = Path("data/processed")
PANEL = PROC / "country_panel.csv"
OUT = PROC / "residuals.csv"

CORE = ["log_ipv4_addresses", "log_gdp_per_capita_usd", "log_population"]


def load():
    df = pd.read_csv(PANEL)
    df = df.dropna(subset=CORE)
    print(f"countries in estimation sample {len(df)}\n")
    return df


def fit(df, formula, label):
    """Fit one OLS model with heteroskedasticity-robust standard errors."""
    model = smf.ols(formula, data=df).fit(cov_type="HC3")
    print("=" * 70)
    print(label)
    print(formula)
    print("=" * 70)
    print(f"n {int(model.nobs)}   R2 {model.rsquared:.3f}   adj R2 {model.rsquared_adj:.3f}")
    print()
    table = pd.DataFrame({
        "coef": model.params,
        "std_err": model.bse,
        "t": model.tvalues,
        "p": model.pvalues,
    })
    print(table.round(4).to_string())
    print()
    return model


def constant_returns_test(model):
    """Test whether address space scales one-for-one with population."""
    if "log_population" not in model.params.index:
        return
    test = model.t_test("log_population = 1")
    coef = model.params["log_population"]
    pval = float(np.ravel(test.pvalue)[0])
    print(f"population elasticity {coef:.3f}")
    print(f"test of elasticity equal to one, p {pval:.4f}")
    if pval < 0.05:
        direction = "more" if coef > 1 else "less"
        print(f"rejected, address space scales {direction} than proportionally with population")
    else:
        print("not rejected, consistent with proportional scaling")
    print()


def main():
    df = load()

    print("correlation of log per-capita footprint with log income")
    sub = df.dropna(subset=["ipv4_per_capita"])
    r = np.corrcoef(np.log(sub["ipv4_per_capita"]), sub["log_gdp_per_capita_usd"])[0, 1]
    print(f"r {r:.3f}   n {len(sub)}\n")

    m1 = fit(df,
             "log_ipv4_addresses ~ log_gdp_per_capita_usd + log_population",
             "Model 1  baseline")
    constant_returns_test(m1)

    have_inet = df["internet_users_pct"].notna()
    m2 = fit(df[have_inet],
             "log_ipv4_addresses ~ log_gdp_per_capita_usd + log_population + internet_users_pct",
             "Model 2  adding internet penetration")

    m3 = fit(df,
             "log_asn_count ~ log_gdp_per_capita_usd + log_population",
             "Model 3  ASN count as outcome")

    df = df.copy()
    df["resid_ipv4"] = m1.resid
    df["fitted_ipv4"] = m1.fittedvalues

    keep = ["iso2", "country_name", "region", "income_group",
            "gdp_per_capita_usd", "population", "ipv4_addresses",
            "log_ipv4_addresses", "fitted_ipv4", "resid_ipv4"]
    keep = [c for c in keep if c in df.columns]
    df[keep].sort_values("resid_ipv4", ascending=False).to_csv(OUT, index=False)
    print(f"residuals written to {OUT}\n")

    show = [c for c in ["iso2", "country_name", "gdp_per_capita_usd",
                        "ipv4_addresses", "resid_ipv4"] if c in df.columns]
    print("most over-provisioned relative to income and population")
    print(df.nlargest(12, "resid_ipv4")[show].to_string(index=False))
    print("\nmost under-provisioned relative to income and population")
    print(df.nsmallest(12, "resid_ipv4")[show].to_string(index=False))

    if "region" in df.columns:
        print("\nmean residual by region")
        print(df.groupby("region")["resid_ipv4"].agg(["mean", "count"]).round(3).to_string())

    if "income_group" in df.columns:
        print("\nmean residual by income group")
        print(df.groupby("income_group")["resid_ipv4"].agg(["mean", "count"]).round(3).to_string())


if __name__ == "__main__":
    main()