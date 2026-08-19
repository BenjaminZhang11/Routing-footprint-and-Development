"""Generate the three headline figures for the project.

Figure 1 is the main result, latency against income after monitor effects are
absorbed. Figure 2 is the announcement ratio against income, which is flat under
a linear fit and curved under a quadratic, with offshore centres marked. Figure
3 compares the three stages of the hierarchy across income groups.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

PROC = Path("data/processed")
FIGS = Path("figures")

EFFECTS = PROC / "country_effects.csv"
OFC_PANEL = PROC / "gap_panel_ofc.csv"

MIN_TRACES = 100

LABEL_LOW_LATENCY = {"GB", "US", "CH", "DE", "NL", "SE", "JP", "KR"}
LABEL_HIGH_LATENCY = {"CN", "MG", "MW", "CM", "MM", "BO", "NP", "ET"}
LABEL_RATIO = {"SC", "SG", "CH", "LU", "IE", "CM", "SD", "MW", "CD", "EG", "BR"}

plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 200,
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
})


def label_points(ax, df, xcol, ycol, codes, dx=0.04, dy=0.006):
    for _, r in df[df["iso2"].isin(codes)].iterrows():
        ax.annotate(r["iso2"], (r[xcol], r[ycol]),
                    xytext=(r[xcol] + dx, r[ycol] + dy),
                    fontsize=8, color="#333333")


def figure_latency(df):
    d = df[df["traces"] >= MIN_TRACES].dropna(
        subset=["log_gdp_pc", "rtt_effect"]).copy()

    m = smf.ols("rtt_effect ~ log_gdp_pc + log_pop", data=d).fit(cov_type="HC3")
    slope = m.params["log_gdp_pc"]
    pct = (np.exp(slope * np.log(2)) - 1) * 100

    fig, ax = plt.subplots(figsize=(7.2, 5))
    sizes = np.clip(np.sqrt(d["traces"]) * 1.6, 12, 260)
    ax.scatter(d["log_gdp_pc"], d["rtt_effect"], s=sizes,
               alpha=0.55, color="#2a6f8e", edgecolor="white", linewidth=0.6)

    xs = np.linspace(d["log_gdp_pc"].min(), d["log_gdp_pc"].max(), 100)
    simple = smf.ols("rtt_effect ~ log_gdp_pc", data=d).fit()
    ax.plot(xs, simple.params["Intercept"] + simple.params["log_gdp_pc"] * xs,
            color="#c1440e", linewidth=2)

    label_points(ax, d, "log_gdp_pc", "rtt_effect",
                 LABEL_LOW_LATENCY | LABEL_HIGH_LATENCY)

    ax.axhline(0, color="#888888", linewidth=0.8, linestyle="--")
    ax.set_xlabel("log GDP per capita, current US dollars")
    ax.set_ylabel("adjusted latency effect, log ms")
    ax.set_title("Richer countries sit closer to the network core", loc="left",
                 fontsize=12, weight="bold")
    ax.text(0.02, 0.03,
            f"n = {int(m.nobs)} countries   R\u00b2 = {m.rsquared:.2f}\n"
            f"doubling income per capita, {pct:.0f}% latency\n"
            f"monitor fixed effects absorbed; point size is trace count",
            transform=ax.transAxes, fontsize=8, color="#444444", va="bottom")

    fig.tight_layout()
    fig.savefig(FIGS / "01_latency_income.png")
    plt.close(fig)
    print(f"figure 1 written, n={int(m.nobs)}, slope={slope:.3f}")


def figure_ratio(df):
    d = df[(df["announcement_ratio"] > 0) & (df["announcement_ratio"] < 1)].copy()
    d = d.dropna(subset=["log_gdp_pc", "announcement_ratio"])
    d["log_gdp_pc_sq"] = d["log_gdp_pc"] ** 2

    lin = smf.ols("announcement_ratio ~ log_gdp_pc", data=d).fit()
    quad = smf.ols("announcement_ratio ~ log_gdp_pc + log_gdp_pc_sq",
                   data=d).fit(cov_type="HC3")

    fig, ax = plt.subplots(figsize=(7.2, 5))
    ofc = d["ofc"] == 1 if "ofc" in d.columns else pd.Series(False, index=d.index)

    ax.scatter(d.loc[~ofc, "log_gdp_pc"], d.loc[~ofc, "announcement_ratio"],
               s=34, alpha=0.55, color="#2a6f8e", edgecolor="white",
               linewidth=0.5, label="other countries")
    ax.scatter(d.loc[ofc, "log_gdp_pc"], d.loc[ofc, "announcement_ratio"],
               s=60, alpha=0.9, color="#c1440e", edgecolor="white",
               linewidth=0.6, marker="D", label="offshore financial centre")

    xs = np.linspace(d["log_gdp_pc"].min(), d["log_gdp_pc"].max(), 200)
    ax.plot(xs, lin.params["Intercept"] + lin.params["log_gdp_pc"] * xs,
            color="#888888", linewidth=1.6, linestyle="--", label="linear fit")
    ax.plot(xs,
            quad.params["Intercept"] + quad.params["log_gdp_pc"] * xs
            + quad.params["log_gdp_pc_sq"] * xs ** 2,
            color="#1b7837", linewidth=2.2, label="quadratic fit")

    label_points(ax, d, "log_gdp_pc", "announcement_ratio", LABEL_RATIO,
                 dx=0.05, dy=0.012)

    ax.set_xlabel("log GDP per capita, current US dollars")
    ax.set_ylabel("share of delegated space announced")
    ax.set_title("The poorest and the richest both announce less, for opposite reasons",
                 loc="left", fontsize=12, weight="bold")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.text(0.02, 0.04,
            f"n = {int(quad.nobs)}   linear R\u00b2 = {lin.rsquared:.3f}   "
            f"quadratic R\u00b2 = {quad.rsquared:.3f}\n"
            f"quadratic term p = {quad.pvalues['log_gdp_pc_sq']:.4f}",
            transform=ax.transAxes, fontsize=8, color="#444444", va="bottom")

    fig.tight_layout()
    fig.savefig(FIGS / "02_announcement_ratio.png")
    plt.close(fig)
    print(f"figure 2 written, quadratic p={quad.pvalues['log_gdp_pc_sq']:.4f}")


def figure_stages(df):
    order = ["Low income", "Lower middle income",
             "Upper middle income", "High income"]
    d = df[df["income_group"].isin(order)].copy()
    g = d.groupby("income_group").agg(
        announced=("announcement_ratio", "mean"),
        reached=("raw_reach", "mean"),
        latency=("raw_median_rtt", "mean"),
    ).reindex(order)

    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    short = ["Low", "Lower mid", "Upper mid", "High"]
    color = "#2a6f8e"

    panels = [
        ("announced", "share of delegated space announced", "Announced"),
        ("reached", "share of traces reaching the country", "Reachable"),
        ("latency", "median latency, ms", "Latency"),
    ]
    for ax, (col, ylab, title) in zip(axes, panels):
        vals = g[col].values
        ax.bar(short, vals, color=color, alpha=0.85, width=0.62)
        ax.set_title(title, fontsize=11, weight="bold", loc="left")
        ax.set_ylabel(ylab, fontsize=9)
        ax.tick_params(axis="x", labelsize=9)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.2f}" if v < 10 else f"{v:.0f}",
                    ha="center", va="bottom", fontsize=8, color="#333333")
        ax.margins(y=0.18)

    fig.suptitle("Each stage of the hierarchy, by income group",
                 fontsize=12, weight="bold", x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(FIGS / "03_stages_by_income.png")
    plt.close(fig)
    print("figure 3 written")


def main():
    FIGS.mkdir(exist_ok=True)

    eff = pd.read_csv(EFFECTS)
    if "raw_reach" not in eff.columns and "reach_country" in eff.columns:
        eff["raw_reach"] = eff["reach_country"]

    ofc = pd.read_csv(OFC_PANEL)
    if "log_gdp_pc" not in ofc.columns:
        ofc["log_gdp_pc"] = np.log(ofc["gdp_per_capita_usd"])

    # Pull across anything figure 3 needs that the effects file lacks.
    for col in ["announcement_ratio", "income_group"]:
        if col not in eff.columns and col in ofc.columns:
            eff = eff.merge(ofc[["iso2", col]], on="iso2", how="left")

    figure_latency(eff)
    figure_stages(eff)
    figure_ratio(ofc)

    print(f"\nfigures written to {FIGS}/")
    for f in sorted(FIGS.glob("*.png")):
        print(f"  {f.name}  {f.stat().st_size/1000:.0f} KB")


if __name__ == "__main__":
    main()