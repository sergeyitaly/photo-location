#!/usr/bin/env python3
"""
Build a StreetCLIP gazetteer JSON from GeoNames dumps (CC BY 4.0 GeoNames).

Downloads (manual):
  https://download.geonames.org/export/dump/cities15000.zip  (>= 15k population)
  https://download.geonames.org/export/dump/countryInfo.txt

Example:
  unzip cities15000.zip
  python scripts/build_streetclip_gazetteer.py \\
    --cities cities15000.txt \\
    --countries countryInfo.txt \\
    --out backend/app/data/generated/streetclip_gazetteer_cities15000.json

Output: JSON array of {city, country, lat, lon, pop} suitable for STREETCLIP_GAZETTEER_PATH.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List


def load_country_names(path: Path) -> Dict[str, str]:
    """ISO 3166-1 alpha-2 -> English country name (GeoNames countryInfo.txt)."""
    out: Dict[str, str] = {}
    raw = path.read_text(encoding="utf-8", errors="replace")
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 5:
            continue
        iso2, country = parts[0].strip().upper(), parts[4].strip()
        if len(iso2) == 2 and country:
            out[iso2] = country
    return out


def iter_city_rows(cities_path: Path, countries: Dict[str, str]) -> List[dict]:
    rows: List[dict] = []
    raw = cities_path.read_text(encoding="utf-8", errors="replace")
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        p = line.split("\t")
        if len(p) < 15:
            continue
        name = p[1].strip()
        lat_s, lon_s = p[4], p[5]
        cc = p[8].strip().upper()
        pop_s = p[14]
        if not name or len(cc) != 2:
            continue
        country = countries.get(cc)
        if not country:
            country = cc
        try:
            lat, lon = float(lat_s), float(lon_s)
        except ValueError:
            continue
        try:
            pop = int(pop_s) if pop_s else 0
        except ValueError:
            pop = 0
        row = {"city": name, "country": country, "lat": lat, "lon": lon}
        if pop > 0:
            row["pop"] = pop
        rows.append(row)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="GeoNames → StreetCLIP gazetteer JSON")
    ap.add_argument("--cities", required=True, type=Path, help="Path to cities*.txt from GeoNames")
    ap.add_argument("--countries", required=True, type=Path, help="Path to countryInfo.txt")
    ap.add_argument("--out", required=True, type=Path, help="Output .json path")
    args = ap.parse_args()

    if not args.cities.is_file():
        print(f"cities file not found: {args.cities}", file=sys.stderr)
        return 1
    if not args.countries.is_file():
        print(f"countryInfo not found: {args.countries}", file=sys.stderr)
        return 1

    countries = load_country_names(args.countries)
    rows = iter_city_rows(args.cities, countries)
    if not rows:
        print("No rows parsed; check file format.", file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} rows → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
