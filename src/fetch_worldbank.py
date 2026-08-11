"""Fetch World Bank development indicators into a per-country table."""

import json
import time
import urllib.request
from pathlib import Path

import pandas as pd

OUT = Path("data/processed")
BASE = "https://api.worldbank.org/v2"
YEARS = "2010:2025"

INDICATORS = {
    "NY.GDP.MKTP.CD": "gdp_usd",
    "NY.GDP.PCAP.CD": "gdp_per_capita_usd",
    "SP.POP.TOTL": "population",
    "IT.NET.USER.ZS": "internet_users_pct",
}


def get_json(url, tries=3):
    """Fetch a URL and return parsed JSON, retrying on transient failure."""
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            if attempt == tries - 1:
                raise
            print(f"  retry {attempt + 1} after {exc}")
            time.sleep(2)


def real_countries():
    """Return ISO2 codes for actual countries, excluding World Bank aggregates."""
    url = f"{BASE}/country?format=json&per_page=400"
    payload = get_json(url)
    rows = payload[1]
    valid = {}
    for r in rows:
        if r.get("region", {}).get("id") == "NA":
            continue
        iso2 = (r.get("iso2Code") or "").strip().upper()
        if not iso2:
            continue
        valid[iso2] = {
            "iso3": r.get("id", ""),
            "country_name": r.get("name", ""),
            "region": r.get("region", {}).get("value", ""),
            "income_group": r.get("incomeLevel", {}).get("value", ""),
        }
    return valid


def fetch_indicator(code, name, valid):
    """Fetch one indicator and keep the most recent non-null year per country."""
    url = (
        f"{BASE}/country/all/indicator/{code}"
        f"?format=json&per_page=20000&date={YEARS}"
    )
    payload = get_json(url)
    if len(payload) < 2 or payload[1] is None:
        raise RuntimeError(f"no data returned for {code}")

    rows = []
    for r in payload[1]:
        if r.get("value") is None:
            continue
        iso2 = (r.get("country", {}).get("id") or "").strip().upper()
        if iso2 not in valid:
            continue
        rows.append({"iso2": iso2, "year": int(r["date"]), name: r["value"]})

    df = pd.DataFrame(rows)
    df = df.sort_values("year").groupby("iso2", as_index=False).last()
    df = df.rename(columns={"year": f"{name}_year"})
    print(f"{name:24s} {len(df):>4} countries")
    return df


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    print("fetching country list")
    valid = real_countries()
    print(f"{len(valid)} countries after removing aggregates\n")

    meta = pd.DataFrame([
        {"iso2": k, **v} for k, v in valid.items()
    ])

    merged = meta
    for code, name in INDICATORS.items():
        df = fetch_indicator(code, name, valid)
        merged = merged.merge(df, on="iso2", how="left")

    merged = merged.sort_values("gdp_usd", ascending=False)
    dest = OUT / "worldbank_country.csv"
    merged.to_csv(dest, index=False)

    print(f"\nwritten to {dest}")
    print(f"rows {len(merged)}")
    print("\nmissing values per column")
    print(merged.isna().sum().to_string())
    print("\ntop rows")
    cols = ["iso2", "country_name", "gdp_usd", "gdp_per_capita_usd",
            "population", "internet_users_pct"]
    print(merged[cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()