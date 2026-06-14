"""
Download GeoNames city dumps + countryInfo, emit StreetCLIP gazetteer JSON.

GeoNames data is CC BY 4.0 — see https://www.geonames.org/
"""

from __future__ import annotations

import json
import logging
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.config import Settings

logger = logging.getLogger(__name__)

GEONAMES_DUMP_BASE = "https://download.geonames.org/export/dump/"
COUNTRY_INFO_NAME = "countryInfo.txt"

# Official zips (population thresholds in GeoNames naming).
GEONAMES_CITY_DUMPS: Dict[str, Tuple[str, str]] = {
    "cities1000": ("cities1000.zip", "All cities with population ≥ 1,000"),
    "cities5000": ("cities5000.zip", "All cities with population ≥ 5,000"),
    "cities15000": ("cities15000.zip", "All cities with population ≥ 15,000"),
}


def _app_data_generated(settings: Settings) -> Path:
    override = (getattr(settings, "gazetteer_data_dir", None) or "").strip()
    if override:
        return Path(override).expanduser().resolve()
    # backend/app/data/generated
    return Path(__file__).resolve().parents[1] / "data" / "generated"


def geonames_cache_dir(settings: Settings) -> Path:
    d = _app_data_generated(settings) / "geonames_cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_country_names(path: Path) -> Dict[str, str]:
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


def _download(url: str, dest: Path, timeout: int = 180) -> None:
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "PhotoGeolocation/1.0 (GeoNames gazetteer build)"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        dest.write_bytes(resp.read())


def ensure_country_info(settings: Settings) -> Path:
    cache = geonames_cache_dir(settings)
    dest = cache / COUNTRY_INFO_NAME
    if not dest.is_file() or dest.stat().st_size < 100:
        url = GEONAMES_DUMP_BASE + COUNTRY_INFO_NAME
        logger.info("Downloading %s", url)
        _download(url, dest)
    return dest


def ensure_city_zip(settings: Settings, dump_key: str) -> Path:
    if dump_key not in GEONAMES_CITY_DUMPS:
        raise ValueError(f"Unknown dump_key: {dump_key}")
    fname = GEONAMES_CITY_DUMPS[dump_key][0]
    cache = geonames_cache_dir(settings)
    dest = cache / fname
    if not dest.is_file() or dest.stat().st_size < 1000:
        url = GEONAMES_DUMP_BASE + fname
        logger.info("Downloading %s (~may be large)", url)
        _download(url, dest, timeout=600)
    return dest


def _read_cities_txt_from_zip(zip_path: Path) -> str:
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = [n for n in zf.namelist() if n.endswith(".txt") and not n.startswith("__")]
        if not names:
            raise ValueError(f"No .txt inside {zip_path}")
        # Prefer canonical cities*.txt name
        names.sort(key=lambda n: (0 if "cities" in n.lower() else 1, len(n)))
        return zf.read(names[0]).decode("utf-8", errors="replace")


def parse_cities_text(
    raw: str,
    countries: Dict[str, str],
    filter_cc: Optional[str],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    target = filter_cc.upper() if filter_cc else None
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
        if target and cc != target:
            continue
        country = countries.get(cc) or cc
        try:
            lat, lon = float(lat_s), float(lon_s)
        except ValueError:
            continue
        try:
            pop = int(pop_s) if pop_s else 0
        except ValueError:
            pop = 0
        row: Dict[str, Any] = {"city": name, "country": country, "lat": lat, "lon": lon}
        if pop > 0:
            row["pop"] = pop
        rows.append(row)
    return rows


def list_countries(settings: Settings) -> List[Dict[str, str]]:
    """ISO2 + English name from cached countryInfo."""
    path = ensure_country_info(settings)
    names = load_country_names(path)
    return [{"code": k, "name": v} for k, v in sorted(names.items(), key=lambda kv: kv[1].lower())]


def build_gazetteer_json(
    settings: Settings,
    *,
    dump_key: str,
    country_iso: str,
) -> Dict[str, Any]:
    """
    Download/cache GeoNames zip + countryInfo, parse, write JSON under app/data/generated.

    country_iso: \"ALL\" / \"\" → worldwide rows from chosen dump; else ISO-3166-1 alpha-2.
    """
    dump_key = dump_key.strip().lower()
    if dump_key not in GEONAMES_CITY_DUMPS:
        raise ValueError(f"dump must be one of: {', '.join(GEONAMES_CITY_DUMPS)}")

    cc_raw = (country_iso or "").strip().upper()
    if not cc_raw or cc_raw in ("ALL", "*", "WORLD"):
        filter_cc: Optional[str] = None
        slug = "world"
    else:
        if len(cc_raw) != 2 or not cc_raw.isalpha():
            raise ValueError("country_iso must be two letters or ALL")
        filter_cc = cc_raw
        slug = cc_raw.lower()

    country_path = ensure_country_info(settings)
    zip_path = ensure_city_zip(settings, dump_key)
    countries_map = load_country_names(country_path)

    raw_txt = _read_cities_txt_from_zip(zip_path)
    rows = parse_cities_text(raw_txt, countries_map, filter_cc)
    if not rows:
        raise ValueError("No city rows after parse/filter — check country_iso or dump.")

    out_dir = _app_data_generated(settings)
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = f"streetclip_gazetteer_{dump_key}_{slug}.json"
    out_path = out_dir / filename
    out_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    from app.data.gazetteer_loader import clear_gazetteer_json_cache

    clear_gazetteer_json_cache()

    rel_for_env = f"app/data/generated/{filename}"
    return {
        "ok": True,
        "rows": len(rows),
        "filename": filename,
        "relative_path": rel_for_env,
        "absolute_path": str(out_path.resolve()),
        "dump_key": dump_key,
        "country_iso": slug.upper() if slug != "world" else "ALL",
        "license_note": "GeoNames CC BY 4.0 — attribute geonames.org when redistributing.",
    }


def resolve_safe_gazetteer_download(settings: Settings, filename: str) -> Path:
    """Resolve a safe path under generated gazetteer dir for GET download."""
    if "/" in filename or "\\" in filename or ".." in filename:
        raise ValueError("invalid filename")
    if not filename.startswith("streetclip_gazetteer_") or not filename.endswith(".json"):
        raise ValueError("invalid filename")
    gen = _app_data_generated(settings)
    path = (gen / filename).resolve()
    try:
        path.relative_to(gen.resolve())
    except ValueError as e:
        raise ValueError("path escape") from e
    return path
