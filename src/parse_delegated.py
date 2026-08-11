"""Parse RIR delegated extended statistics files into per-country resource counts."""

from pathlib import Path
import pandas as pd

RAW = Path("data/raw")
OUT = Path("data/processed")

COLUMNS = ["registry", "cc", "type", "start", "value", "date", "status", "opaque_id"]
VALID_TYPES = {"asn", "ipv4", "ipv6"}
DELEGATED_STATUS = {"allocated", "assigned"}

FILES = [
    "delegated-afrinic-extended-latest",
    "delegated-apnic-extended-latest",
    "delegated-arin-extended-latest",
    "delegated-lacnic-extended-latest",
    "delegated-ripencc-extended-latest",
]


def parse_delegated(path):
    """Read one delegated file into a dataframe of resource records."""
    rows = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.rstrip("\n").split("|")
            if len(parts) < 7:
                continue
            rtype = parts[2].lower()
            if rtype not in VALID_TYPES:
                continue
            cc = parts[1].strip().upper()
            if cc in ("", "*", "ZZ"):
                continue
            rec = dict(zip(COLUMNS, parts[:8]))
            rec["type"] = rtype
            rec["cc"] = cc
            rec["status"] = parts[6].strip().lower()
            rows.append(rec)
    return pd.DataFrame(rows, columns=COLUMNS)


def load_all():
    frames = []
    for name in FILES:
        path = RAW / name
        if not path.exists():
            raise FileNotFoundError(f"missing {path}")
        df = parse_delegated(path)
        print(f"{name:45s} {len(df):>8,} records")
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df


def aggregate(df):
    """Collapse to one row per country."""
    d = df[df["status"].isin(DELEGATED_STATUS)]

    ipv4 = d[d["type"] == "ipv4"].groupby("cc")["value"].sum().rename("ipv4_addresses")
    ipv4_blocks = d[d["type"] == "ipv4"].groupby("cc").size().rename("ipv4_blocks")
    asn = d[d["type"] == "asn"].groupby("cc")["value"].sum().rename("asn_count")
    ipv6_blocks = d[d["type"] == "ipv6"].groupby("cc").size().rename("ipv6_blocks")

    out = pd.concat([ipv4, ipv4_blocks, asn, ipv6_blocks], axis=1)
    out = out.fillna(0).astype("int64").reset_index()
    out = out.rename(columns={"cc": "iso2"})
    return out.sort_values("ipv4_addresses", ascending=False)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_all()
    print(f"\ntotal records {len(df):,}")
    print(f"statuses seen {sorted(df['status'].unique())}")

    out = aggregate(df)
    dest = OUT / "rir_country.csv"
    out.to_csv(dest, index=False)

    print(f"\ncountries {len(out)}")
    print(f"written to {dest}\n")
    print(out.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
