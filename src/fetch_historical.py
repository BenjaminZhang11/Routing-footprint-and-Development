"""Download one RIR delegated snapshot per year from the registry archives.

Each registry lays out its history differently, so this lists each year folder,
picks the snapshot closest to a target date, and stores everything gzipped under
data/raw/historical with one consistent naming scheme.

Older files use the non-extended format, which still carries registry, country,
type, start, value, date, and status in the first seven fields. That is all the
parser reads, so extended and non-extended snapshots are interchangeable here.
"""

import gzip
import re
import ssl
import sys
import urllib.request
from datetime import date
from pathlib import Path

OUT = Path("data/raw/historical")

START_YEAR = 2003
END_YEAR = 2026
TARGET_MONTH = 7
TARGET_DAY = 1

REGISTRIES = {
    "ripencc": "https://ftp.ripe.net/pub/stats/ripencc/{year}/",
    "apnic": "https://ftp.apnic.net/stats/apnic/{year}/",
    "arin": "https://ftp.arin.net/pub/stats/arin/archive/{year}/",
    "lacnic": "https://ftp.lacnic.net/pub/stats/lacnic/archive/{year}/",
    "afrinic": "https://ftp.afrinic.net/pub/stats/afrinic/{year}/",
}

CTX = ssl.create_default_context()


def get(url, timeout=90):
    req = urllib.request.Request(url, headers={"User-Agent": "escalate-capstone"})
    with urllib.request.urlopen(req, timeout=timeout, context=CTX) as resp:
        return resp.read()


def list_snapshots(base_url, rir):
    """Return {date_string: filename} for usable snapshots in one year folder."""
    try:
        html = get(base_url, timeout=60).decode("utf-8", errors="replace")
    except Exception as exc:
        print(f"    listing failed  {exc}")
        return {}

    found = {}
    pattern = re.compile(
        rf"delegated-{rir}-(extended-)?(\d{{8}})(\.gz|\.bz2)?$")

    for m in re.finditer(r'href="([^"]+)"', html):
        name = m.group(1)
        if ".md5" in name or ".asc" in name:
            continue
        mm = pattern.fullmatch(name)
        if not mm:
            continue
        is_extended = bool(mm.group(1))
        stamp = mm.group(2)
        prev = found.get(stamp)
        if prev is None or (is_extended and not prev[1]):
            found[stamp] = (name, is_extended)
    return found


def pick_nearest(found, year):
    """Choose the snapshot closest to the target date within a year."""
    if not found:
        return None
    target = date(year, TARGET_MONTH, TARGET_DAY)

    def distance(stamp):
        try:
            d = date(int(stamp[:4]), int(stamp[4:6]), int(stamp[6:8]))
        except ValueError:
            return 10 ** 6
        return abs((d - target).days)

    best = min(found, key=distance)
    return best, found[best][0]


def decompress(raw, name):
    if name.endswith(".gz"):
        return gzip.decompress(raw)
    if name.endswith(".bz2"):
        import bz2
        return bz2.decompress(raw)
    return raw


def fetch_year(rir, template, year):
    dest = OUT / f"{rir}-{year}.gz"
    if dest.exists() and dest.stat().st_size > 1000:
        print(f"    already have {dest.name}")
        return True

    base = template.format(year=year)
    found = list_snapshots(base, rir)
    choice = pick_nearest(found, year)
    if choice is None:
        print(f"    no snapshot found for {year}")
        return False

    stamp, name = choice
    url = base + name
    try:
        raw = get(url)
        text = decompress(raw, name)
    except Exception as exc:
        print(f"    download failed  {exc}")
        return False

    lines = text.count(b"\n")
    if lines < 100:
        print(f"    suspiciously small, {lines} lines, skipping")
        return False

    OUT.mkdir(parents=True, exist_ok=True)
    with gzip.open(dest, "wb") as f:
        f.write(text)
    print(f"    {stamp}  {lines:>7,} lines  ->  {dest.name}")
    return True


def main():
    years = range(START_YEAR, END_YEAR + 1)
    summary = {}

    for rir, template in REGISTRIES.items():
        print(f"\n{rir}")
        got = 0
        for year in years:
            print(f"  {year}")
            if fetch_year(rir, template, year):
                got += 1
        summary[rir] = got

    print("\nsnapshots retrieved")
    for rir, n in summary.items():
        print(f"  {rir:10s} {n:>3} of {len(list(years))}")

    total = sum(f.stat().st_size for f in OUT.glob("*.gz"))
    print(f"\nfiles on disk {len(list(OUT.glob('*.gz')))}")
    print(f"total size {total / 1e6:.1f} MB")


if __name__ == "__main__":
    sys.exit(main())