"""Build a per-trace table from Ark data, mapping traces to delegation blocks.

Task 1 builds searchable country lookups from the RIR delegated files for IPv4
ranges, IPv6 ranges, and AS number ranges across all five registries.

Task 2 reads each Ark trace and records the last router that responded to a TTL
probe, then maps both the trace destination and that last responding hop into
the delegation blocks from task 1. A trace does not need to reach its
destination to be informative. If the last responding hop falls inside the
destination country's delegated space, the path reached that country's network
even when the final host stayed silent, which is the common case.
"""

import gzip
import ipaddress
import json
import subprocess
from bisect import bisect_right
from pathlib import Path

import pandas as pd

RAW = Path("data/raw")
PROC = Path("data/processed")
WARTS_DIR = RAW / "ark"
OUT = PROC / "trace_table.csv"

DELEGATED_FILES = [
    "delegated-afrinic-extended-latest",
    "delegated-apnic-extended-latest",
    "delegated-arin-extended-latest",
    "delegated-lacnic-extended-latest",
    "delegated-ripencc-extended-latest",
]
DELEGATED_STATUS = {"allocated", "assigned"}


class RangeLookup:
    """Sorted non-overlapping ranges supporting binary-search containment."""

    def __init__(self, rows):
        rows.sort()
        self.starts = [r[0] for r in rows]
        self.ends = [r[1] for r in rows]
        self.ccs = [r[2] for r in rows]

    def __len__(self):
        return len(self.starts)

    def find(self, value):
        i = bisect_right(self.starts, value) - 1
        if i >= 0 and value < self.ends[i]:
            return self.ccs[i]
        return None


def build_lookups():
    """Task 1. Parse every registry for IPv4, IPv6, and AS number delegations."""
    v4, v6, asn = [], [], []

    for name in DELEGATED_FILES:
        path = RAW / name
        if not path.exists():
            raise FileNotFoundError(f"missing {path}")
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                parts = line.rstrip("\n").split("|")
                if len(parts) < 7:
                    continue
                rtype = parts[2].lower()
                cc = parts[1].strip().upper()
                if cc in ("", "*", "ZZ"):
                    continue
                if parts[6].strip().lower() not in DELEGATED_STATUS:
                    continue

                try:
                    if rtype == "ipv4":
                        start = int(ipaddress.IPv4Address(parts[3]))
                        v4.append((start, start + int(parts[4]), cc))
                    elif rtype == "ipv6":
                        start = int(ipaddress.IPv6Address(parts[3]))
                        size = 1 << (128 - int(parts[4]))
                        v6.append((start, start + size, cc))
                    elif rtype == "asn":
                        start = int(parts[3])
                        asn.append((start, start + int(parts[4]), cc))
                except (ValueError, ipaddress.AddressValueError):
                    continue

    lookups = {
        "ipv4": RangeLookup(v4),
        "ipv6": RangeLookup(v6),
        "asn": RangeLookup(asn),
    }
    print("task 1, delegation ranges loaded")
    for k, v in lookups.items():
        print(f"  {k:5s} {len(v):>8,} ranges")
    return lookups


def last_responding_hop(rec):
    """Return the furthest hop that answered, by probe TTL."""
    hops = [h for h in rec.get("hops", []) if h.get("addr")]
    if not hops:
        return None
    return max(hops, key=lambda h: h.get("probe_ttl", 0))


def iter_traces(path):
    proc = subprocess.Popen(
        ["sc_warts2json", str(path)],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
    )
    for line in proc.stdout:
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("type") == "trace":
            yield rec
    proc.wait()


def to_int(addr):
    try:
        return int(ipaddress.IPv4Address(addr))
    except ipaddress.AddressValueError:
        return None


def build_table(lookups):
    """Task 2. One row per trace, with destination and last hop attributed."""
    v4 = lookups["ipv4"]
    files = sorted(WARTS_DIR.glob("*.warts.gz"))
    if not files:
        raise FileNotFoundError(f"no warts files in {WARTS_DIR}")
    print(f"\nwarts files {len(files)}")

    rows = []
    for path in files:
        n = 0
        for rec in iter_traces(path):
            dst = rec.get("dst")
            dst_int = to_int(dst) if dst else None
            if dst_int is None:
                continue

            hop = last_responding_hop(rec)
            hop_addr = hop.get("addr") if hop else None
            hop_int = to_int(hop_addr) if hop_addr else None

            rows.append({
                "monitor": rec.get("monitor", ""),
                "src": rec.get("src", ""),
                "dst": dst,
                "dst_cc": v4.find(dst_int),
                "last_hop_ip": hop_addr,
                "last_hop_cc": v4.find(hop_int) if hop_int is not None else None,
                "last_hop_ttl": hop.get("probe_ttl") if hop else None,
                "rtt_ms": hop.get("rtt") if hop else None,
                "stop_reason": rec.get("stop_reason", ""),
                "hop_count": rec.get("hop_count"),
            })
            n += 1
        print(f"  {path.name:45s} {n:>7,} traces")

    df = pd.DataFrame(rows)
    df["reached_dst_country"] = (
        df["last_hop_cc"].notna()
        & df["dst_cc"].notna()
        & (df["last_hop_cc"] == df["dst_cc"])
    )
    return df


def summarise(df):
    print(f"\ntotal traces {len(df):,}")
    print(f"traces with a responding hop {df['last_hop_ip'].notna().sum():,}")
    print(f"destination attributed to a country {df['dst_cc'].notna().sum():,}")
    print(f"last hop landed in destination country {df['reached_dst_country'].sum():,}")

    known = df[df["dst_cc"].notna()]
    agg = known.groupby("dst_cc").agg(
        traces=("dst", "size"),
        reached=("reached_dst_country", "sum"),
        median_rtt=("rtt_ms", "median"),
    )
    agg["reach_rate"] = (agg["reached"] / agg["traces"]).round(3)
    agg["median_rtt"] = agg["median_rtt"].round(1)
    agg = agg.sort_values("traces", ascending=False)

    dest = PROC / "reach_by_country.csv"
    agg.reset_index().rename(columns={"dst_cc": "iso2"}).to_csv(dest, index=False)
    print(f"\ncountry summary written to {dest}")

    enough = agg[agg["traces"] >= 30]
    print(f"\ncountries with at least 30 traces {len(enough)}")
    print("\nhighest share of traces reaching the destination country")
    print(enough.nlargest(10, "reach_rate").to_string())
    print("\nlowest share")
    print(enough.nsmallest(10, "reach_rate").to_string())


def main():
    lookups = build_lookups()
    df = build_table(lookups)

    PROC.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"\ntrace table written to {OUT}")

    cols = ["monitor", "dst", "dst_cc", "last_hop_ip", "last_hop_cc",
            "rtt_ms", "stop_reason"]
    print("\nsample rows")
    print(df[cols].head(10).to_string(index=False))

    summarise(df)


if __name__ == "__main__":
    main()