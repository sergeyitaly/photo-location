#!/usr/bin/env python3
"""
Regenerate backend/app/data/world_country_clip_centroids.py from REST Countries API.

Usage:
  curl -sS "https://restcountries.com/v3.1/all?fields=name,cca2,latlng" | \\
    python scripts/generate_world_country_clip_centroids.py

Requires network for curl step only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    raw = sys.stdin.read()
    if not raw.strip():
        print("Read JSON from stdin (pipe REST Countries response).", file=sys.stderr)
        return 1
    d = json.loads(raw)
    rows: list[tuple[str, str, float, float]] = []
    for x in d:
        cc = x.get("cca2")
        nm = x.get("name", {}).get("common")
        ll = x.get("latlng")
        if not cc or not nm or not ll or len(ll) < 2:
            continue
        lat, lon = float(ll[0]), float(ll[1])
        nm_esc = nm.replace("\\", "\\\\").replace('"', '\\"')
        rows.append((nm_esc, cc, lat, lon))
    rows.sort(key=lambda r: r[0])

    backend = Path(__file__).resolve().parents[1] / "backend"
    path = backend / "app" / "data" / "world_country_clip_centroids.py"
    lines = [
        '"""Worldwide ISO-level countries for CLIP zero-shot country softmax."""',
        "",
        "from __future__ import annotations",
        "",
        "from typing import List, Tuple",
        "",
        "# English short names + REST Countries latlng centroids.",
        '# Regenerate: curl -s "https://restcountries.com/v3.1/all?fields=name,cca2,latlng" \\',
        "#   | python scripts/generate_world_country_clip_centroids.py",
        "COUNTRY_ENTRIES_WORLDWIDE: List[Tuple[str, float, float]] = [",
    ]
    for nm, cc, lat, lon in rows:
        lines.append(f'    ("{nm}", {lat:.8f}, {lon:.8f}),  # {cc}')
    lines.append("]")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {len(rows)} countries → {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
