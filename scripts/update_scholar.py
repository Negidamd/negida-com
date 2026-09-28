#!/usr/bin/env python3
"""Refresh data/scholar.json from the public Google Scholar profile.

Run weekly by .github/workflows/scholar.yml. Uses only the standard library.
If Scholar blocks the request or the page layout changes, the existing file is
left untouched and the script exits 0, so the site keeps its last good numbers.
"""
import datetime
import json
import pathlib
import re
import sys
import urllib.request

PROFILE = "https://scholar.google.com/citations?user=HURlCI8AAAAJ&hl=en"
OUT = pathlib.Path(__file__).resolve().parent.parent / "data" / "scholar.json"


def fetch(url):
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/141.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
    })
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode("utf-8", "replace")


def parse(page):
    stats = dict((label, (int(total), int(recent))) for label, total, recent in re.findall(
        r'<td class="gsc_rsb_sc1">.*?>([^<]+)</a></td><td class="gsc_rsb_std">(\d+)</td>'
        r'<td class="gsc_rsb_std">(\d+)</td>', page))
    years = [int(y) for y in re.findall(r'<span class="gsc_g_t"[^>]*>(\d{4})</span>', page)]
    counts = [int(v) for v in re.findall(r'<span class="gsc_g_al">(\d+)</span>', page)]
    if not {"Citations", "h-index", "i10-index"} <= stats.keys() or not years or len(counts) > len(years):
        return None
    return {
        "citations": stats["Citations"][0],
        "h_index": stats["h-index"][0],
        "i10_index": stats["i10-index"][0],
        "per_year": [[y, c] for y, c in zip(years[-len(counts):], counts)],
    }


def main():
    try:
        data = parse(fetch(PROFILE))
    except Exception as exc:  # network error, block page, timeout
        print(f"Scholar fetch failed ({exc}); keeping existing data")
        return 0
    if not data or data["citations"] < 1000 or not 10 <= data["h_index"] <= 300:
        print("Could not read metrics (blocked or layout changed); keeping existing data")
        return 0
    old = json.loads(OUT.read_text()) if OUT.exists() else {}
    if all(old.get(k) == data[k] for k in data):
        print("No change")
        return 0
    data = {"updated": datetime.date.today().isoformat(), "source": PROFILE, **data}
    OUT.write_text(json.dumps(data, indent=2) + "\n")
    print(f"Updated: {data['citations']} citations, h-index {data['h_index']}, i10-index {data['i10_index']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
