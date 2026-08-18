"""Compute per-country reachability from CAIDA Ark team-probing traces.

Runs sc_warts2json on a warts file and streams the output, so no intermediate
file is written. Each trace targets one destination in a routed /24. A trace
with stop_reason COMPLETED reached its destination and got a reply; anything
else did not, and stop_reason records why, which matters because "no reply"
can mean no host, a filtering host, or a broken path, not necessarily an
inactive network. Destinations are attributed to countries using the same
delegated blocks as the announced-space script, so reachable, announced, and
delegated space all share one attribution basis.
"""

import ipaddress
import json
import subprocess
from bisect import bisect_right
from collections import defaultdict
from pathlib import Path

import pandas as pd

RAW = Path("data/raw")
PROC = Path("data/processed")

WARTS_DIR = RAW / "ark"
OUT = PROC / "reachability_country.csv"

DELEGATED_FILES = [
    "delegated-afrinic-extended-latest",
    "delegated-apnic-extended-latest",
    "delegated-arin-extended-latest",
    "delegated-lacnic-extended-latest",
    "delegated-ripencc-extended-latest",
]
DELEGATED_STATUS = {"allocated", "assigned"}


def load_delegated_blocks():
    rows = []
    for name in DELEGATED_FILES:
        path = RAW / name
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
    return [r[0] for r in rows], [r[1] for r in rows], [r[2] for r in rows]


def make_lookup(starts, ends, ccs):
    def lookup(ip_int):
        i = bisect_right(starts, ip_int) - 1
        if i >= 0 and ip_int < ends[i]:
            return ccs[i]
        return None
    return lookup


def iter_traces(warts_path):
    """Stream trace records out of a warts file via sc_warts2json."""
    proc = subprocess.Popen(
        ["sc_warts2json", str(warts_path)],
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


def process_file(path, lookup, counts, reasons):
    n = 0
    for rec in iter_traces(path):
        dst = rec.get("dst")
        reason = rec.get("stop_reason", "UNKNOWN")
        if not dst:
            continue
        try:
            ip_int = int(ipaddress.IPv4Address(dst))
        except ipaddress.AddressValueError:
            continue
        cc = lookup(ip_int)
        if cc is None:
            continue
        n += 1
        counts[cc]["total"] += 1
        if reason == "COMPLETED":
            counts[cc]["reached"] += 1
        reasons[cc][reason] += 1
    return n


def main():
    starts, ends, ccs = load_delegated_blocks()
    lookup = make_lookup(starts, ends, ccs)
    print(f"delegated blocks loaded {len(starts):,}")

    files = sorted(WARTS_DIR.glob("*.warts.gz"))
    if not files:
        raise FileNotFoundError(f"no warts files in {WARTS_DIR}")
    print(f"warts files found {len(files)}")

    counts = defaultdict(lambda: {"total": 0, "reached": 0})
    reasons = defaultdict(lambda: defaultdict(int))

    total_traces = 0
    for path in files:
        n = process_file(path, lookup, counts, reasons)
        total_traces += n
        print(f"  {path.name:45s} {n:>7,} attributed traces")

    print(f"\ntotal attributed traces {total_traces:,}")

    rows = []
    for cc, c in counts.items():
        rows.append({
            "iso2": cc,
            "probes_total": c["total"],
            "probes_reached": c["reached"],
            "reachability_rate": c["reached"] / c["total"] if c["total"] else None,
        })
    df = pd.DataFrame(rows).sort_values("probes_total", ascending=False)

    PROC.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"\ncountries with probe data {len(df)}")
    print(f"written to {OUT}\n")

    enough = df[df["probes_total"] >= 20]
    print(f"countries with at least 20 probes {len(enough)}\n")
    print("lowest reachability rate")
    print(enough.nsmallest(12, "reachability_rate").to_string(index=False))
    print("\nhighest reachability rate")
    print(enough.nlargest(12, "reachability_rate").to_string(index=False))

    print("\nstop reasons, top 10 countries by probe count")
    for cc in df.head(10)["iso2"]:
        top_reasons = sorted(reasons[cc].items(), key=lambda x: -x[1])[:4]
        print(f"  {cc}  {top_reasons}")


if __name__ == "__main__":
    main()