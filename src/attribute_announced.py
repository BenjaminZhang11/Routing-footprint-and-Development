"""Attribute announced BGP prefixes to countries using RIR delegated blocks.

Reads the CAIDA RouteViews prefix-to-AS file and maps each announced prefix into
the delegated block that contains it, so announced and delegated address space
share one country-attribution basis and the gap between them is meaningful.

Announced prefixes overlap, because operators commonly announce a covering
prefix alongside more specific subprefixes for traffic engineering. Summing
prefix sizes therefore double counts address space. This script reports the
union of announced ranges per country as the headline measure, and also reports
the naive sum so the size of the overlap is visible.
"""

import gzip
import ipaddress
from bisect import bisect_right
from collections import defaultdict
from pathlib import Path

import pandas as pd

RAW = Path("data/raw")
PROC = Path("data/processed")

PFX2AS = RAW / "routeviews-pfx2as.gz"
OUT = PROC / "announced_country.csv"

DELEGATED_FILES = [
    "delegated-afrinic-extended-latest",
    "delegated-apnic-extended-latest",
    "delegated-arin-extended-latest",
    "delegated-lacnic-extended-latest",
    "delegated-ripencc-extended-latest",
]

DELEGATED_STATUS = {"allocated", "assigned"}


def load_delegated_blocks():
    """Return sorted, non-overlapping IPv4 ranges with their country codes."""
    rows = []
    for name in DELEGATED_FILES:
        path = RAW / name
        if not path.exists():
            raise FileNotFoundError(f"missing {path}")
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                parts = line.rstrip("\n").split("|")
                if len(parts) < 7 or parts[2].lower() != "ipv4":
                    continue
                cc = parts[1].strip().upper()
                if cc in ("", "*", "ZZ"):
                    continue
                if parts[6].strip().lower() not in DELEGATED_STATUS:
                    continue
                try:
                    start = int(ipaddress.IPv4Address(parts[3]))
                    size = int(parts[4])
                except (ValueError, ipaddress.AddressValueError):
                    continue
                rows.append((start, start + size, cc))

    rows.sort()
    starts = [r[0] for r in rows]
    ends = [r[1] for r in rows]
    ccs = [r[2] for r in rows]
    print(f"delegated IPv4 blocks {len(rows):,}")
    return starts, ends, ccs


def make_lookup(starts, ends, ccs):
    """Binary search an address into its containing delegated block."""
    def lookup(ip_int):
        i = bisect_right(starts, ip_int) - 1
        if i >= 0 and ip_int < ends[i]:
            return ccs[i]
        return None
    return lookup


def parse_asns(field):
    """Extract origin ASNs, handling MOAS underscores and AS-set commas."""
    out = set()
    for chunk in field.replace("_", ",").split(","):
        chunk = chunk.strip()
        if chunk.isdigit():
            out.add(int(chunk))
    return out


def union_length(intervals):
    """Total length covered by a set of half-open ranges, counting overlap once."""
    if not intervals:
        return 0
    intervals.sort()
    total = 0
    cur_start, cur_end = intervals[0]
    for start, end in intervals[1:]:
        if start <= cur_end:
            if end > cur_end:
                cur_end = end
        else:
            total += cur_end - cur_start
            cur_start, cur_end = start, end
    total += cur_end - cur_start
    return total


def attribute(lookup):
    intervals = defaultdict(list)
    naive_sum = defaultdict(int)
    prefixes = defaultdict(int)
    origins = defaultdict(set)

    total_pfx = 0
    unattr_pfx = 0

    with gzip.open(PFX2AS, "rt", errors="replace") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            try:
                start = int(ipaddress.IPv4Address(parts[0]))
                plen = int(parts[1])
            except (ValueError, ipaddress.AddressValueError):
                continue
            if not 0 <= plen <= 32:
                continue

            size = 1 << (32 - plen)
            total_pfx += 1

            cc = lookup(start)
            if cc is None:
                unattr_pfx += 1
                continue

            intervals[cc].append((start, start + size))
            naive_sum[cc] += size
            prefixes[cc] += 1
            origins[cc] |= parse_asns(parts[2])

    rows = []
    for cc, ivs in intervals.items():
        rows.append({
            "iso2": cc,
            "announced_addresses": union_length(ivs),
            "announced_addresses_naive": naive_sum[cc],
            "announced_prefixes": prefixes[cc],
            "origin_asns": len(origins[cc]),
        })

    df = pd.DataFrame(rows)
    df["overlap_ratio"] = (
        df["announced_addresses_naive"] / df["announced_addresses"]
    ).round(3)

    union_total = df["announced_addresses"].sum()
    naive_total = df["announced_addresses_naive"].sum()

    print(f"announced prefixes read {total_pfx:,}")
    print(f"unattributed prefixes {unattr_pfx:,} "
          f"({unattr_pfx / max(total_pfx, 1) * 100:.1f} percent)")
    print(f"announced addresses, naive sum {naive_total:,}")
    print(f"announced addresses, union     {union_total:,}")
    print(f"inflation from overlapping announcements {naive_total / union_total:.3f}\n")

    return df.sort_values("announced_addresses", ascending=False)


def main():
    PROC.mkdir(parents=True, exist_ok=True)

    starts, ends, ccs = load_delegated_blocks()
    lookup = make_lookup(starts, ends, ccs)

    df = attribute(lookup)
    df.to_csv(OUT, index=False)
    print(f"countries with announced space {len(df)}")
    print(f"written to {OUT}\n")

    show = ["iso2", "announced_addresses", "announced_prefixes",
            "origin_asns", "overlap_ratio"]
    print(df[show].head(15).to_string(index=False))

    print("\nhighest overlap ratio among countries with substantial space")
    big = df[df["announced_addresses"] > 1_000_000]
    print(big.nlargest(10, "overlap_ratio")[show].to_string(index=False))


if __name__ == "__main__":
    main()