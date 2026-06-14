"""
Global-scale geolocation pipeline (versioned contract).

This stack is designed for **worldwide** use — no single city list can enumerate every settlement.

Stages (vision path):
  1. **Multi-model fusion** — GeoCLIP (globe retrieval) + StreetCLIP (sparse gazetteer softmax) +
     CLIP zero-shot country/landmark cues; optional hybrid reconcile for regional vs capital pins.
  2. **Open-data validation** (optional) — English Wikipedia geosearch + OpenTopoData relief + optional
     CLIP-vs-Wikipedia semantic gate across candidate pins.
  3. **Place naming** — OpenStreetMap Nominatim reverse geocode at the predicted coordinates yields
     city/town/village/hamlet names **anywhere OSM covers**, in the client’s preferred language when set.

Operational notes for production:
  - Host your own **Nominatim** or contract a geocoding provider for throughput; public OSM has strict quotas.
  - Keep **``NOMINATIM_HTTP_USER_AGENT``** truthful (OSM usage policy).
  - Vision accuracy remains bounded; EXIF GPS remains ground truth when present.
"""

PIPELINE_VERSION = "2"
