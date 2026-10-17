"""Bounded public-data clients. No Google imagery or arbitrary URL downloads."""
from __future__ import annotations

import hashlib
import io
import json
import math
import re
import time
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import httpx
import numpy as np
from PIL import Image

HEADERS = {"User-Agent": "RealityForge-Hackathon/0.1 (local educational world generator)"}
OVERPASS = ("https://overpass.private.coffee/api/interpreter", "https://overpass-api.de/api/interpreter")
DEMO_CENTER = (43.6532, -79.3832)


def bbox_dimensions(bbox):
    w, s, e, n = bbox
    return (e - w) * 111320 * math.cos(math.radians((n + s) / 2)), (n - s) * 111320


def validate_bbox(bbox):
    if len(bbox) != 4 or not all(math.isfinite(v) for v in bbox):
        raise ValueError("Bounding box must contain four finite numbers: west, south, east, north.")
    w, s, e, n = bbox
    if not (-180 <= w < e <= 180 and -85 <= s < n <= 85):
        raise ValueError("Choose an ordered bounding box between 85°S and 85°N, without crossing the date line.")
    width, depth = bbox_dimensions(bbox)
    if min(width, depth) < 32 or max(width, depth) > 1010:
        raise ValueError("Each side of the bounding box must be between 32 and 1,000 metres.")
    return bbox


def centered_bbox(lat, lon, size=250):
    dy = size / 111320 / 2
    dx = dy / math.cos(math.radians(lat))
    return [lon - dx, lat - dy, lon + dx, lat + dy]


def parse_location(value: str):
    """Read coordinates explicitly present in a string; never follow short URLs."""
    raw = unquote(value.strip())
    coordinate = r"(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)"
    match = re.fullmatch(coordinate, raw)
    if not match:
        url = urlparse(raw)
        host = (url.hostname or "").lower()
        if url.scheme not in ("https", "http") or not (re.fullmatch(r"(?:www\.|maps\.)?google\.(?:com|ca|co\.uk|com\.au|de|fr|co\.in)", host) or host in ("maps.app.goo.gl", "goo.gl")):
            raise ValueError("Paste latitude, longitude or a full Google Maps URL containing coordinates.")
        # Place coordinates are more precise than the viewport's @lat,lon center.
        place = re.search(r"!3d(-?\d+(?:\.\d+)?)!4d(-?\d+(?:\.\d+)?)", raw)
        match = place or re.search("@" + coordinate, raw)
        if not match:
            params = parse_qs(url.query)
            for key in ("query", "q", "ll", "center", "viewpoint"):
                if key in params:
                    match = re.fullmatch(coordinate, params[key][0])
                    if match:
                        break
    if not match:
        raise ValueError("This link has no coordinates. Open it in Google Maps and copy the full address bar URL, or paste latitude, longitude.")
    lat, lon = map(float, match.groups())
    if not (-85 <= lat <= 85 and -180 <= lon <= 180):
        raise ValueError("Coordinates must be within 85°S–85°N and 180°W–180°E.")
    return {"lat": lat, "lon": lon}


def fetch_osm(bbox, cache_dir: Path):
    validate_bbox(bbox)
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(json.dumps(bbox).encode()).hexdigest()[:24]
    cached = cache_dir / f"osm-{key}.json"
    if cached.exists() and time.time() - cached.stat().st_mtime < 86400:
        return json.loads(cached.read_text(encoding="utf-8"))
    w, s, e, n = bbox
    area = f"({s},{w},{n},{e})"
    selectors = ['way["building"]', 'relation["building"]["type"="multipolygon"]', 'way["highway"]',
                 'way["natural"="water"]', 'relation["natural"="water"]["type"="multipolygon"]',
                 'way["waterway"]', 'way["leisure"="park"]', 'way["landuse"~"forest|grass|recreation_ground"]',
                 'way["natural"="wood"]', 'node["natural"="tree"]']
    query = '[out:json][timeout:35];(' + ''.join(f'{tag}{area};' for tag in selectors) + ');out geom;'
    errors = []
    with httpx.Client(headers=HEADERS, timeout=45, follow_redirects=False) as client:
        for endpoint in OVERPASS:
            try:
                with client.stream("POST", endpoint, data={"data": query}) as response:
                    response.raise_for_status()
                    parts, total = [], 0
                    for chunk in response.iter_bytes():
                        total += len(chunk)
                        if total > 32 * 1024 * 1024:
                            raise ValueError("Map response is too large. Choose a smaller area.")
                        parts.append(chunk)
                data = json.loads(b''.join(parts))
                if data.get("remark"):
                    raise ValueError("Map service could not complete this area: " + data["remark"][:160])
                if not isinstance(data.get("elements"), list) or len(data["elements"]) > 50000:
                    raise ValueError("Map response is invalid or too dense. Choose a smaller area.")
                data["realityforge_source"] = "OpenStreetMap contributors (ODbL), Overpass API"
                cached.write_text(json.dumps(data), encoding="utf-8")
                return data
            except (httpx.HTTPError, ValueError) as exc:
                errors.append(str(exc)[:180])
        # Small-area recovery only, never tile or bulk-download the editing API.
        # OSM includes way nodes; the semantic importer resolves these references.
        if max(bbox_dimensions(bbox)) <= 251:
            try:
                with client.stream("GET", "https://api.openstreetmap.org/api/0.6/map.json",
                                   params={"bbox": ",".join(str(v) for v in bbox)}) as response:
                    response.raise_for_status()
                    parts, total = [], 0
                    for chunk in response.iter_bytes():
                        total += len(chunk)
                        if total > 16 * 1024 * 1024:
                            raise ValueError("Small-area map fallback exceeded its download budget.")
                        parts.append(chunk)
                data = json.loads(b''.join(parts))
                if not isinstance(data.get("elements"), list) or len(data["elements"]) > 50000:
                    raise ValueError("Small-area fallback is too dense.")
                data["realityforge_source"] = "OpenStreetMap contributors (ODbL), small-area OSM API fallback"
                cached.write_text(json.dumps(data), encoding="utf-8")
                return data
            except (httpx.HTTPError, ValueError) as exc:
                errors.append(str(exc)[:180])
    raise RuntimeError("Live map data is unavailable. Retry later or explicitly choose Offline demo. " + " / ".join(errors))


def fetch_elevation(bbox, cache_dir: Path, grid_size=25):
    """Sample public Mapzen Terrarium DEM tiles, ~38m source resolution at equator."""
    w, s, e, n = bbox
    zoom = 12
    cache_dir.mkdir(parents=True, exist_ok=True)

    def pixel(lat, lon):
        scale = 256 * 2 ** zoom
        return ((lon + 180) / 360 * scale,
                (1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * scale)

    while zoom > 8:
        x0, y0 = pixel(n, w)
        x1, y1 = pixel(s, e)
        if (int(x1 // 256) - int(x0 // 256) + 1) * (int(y1 // 256) - int(y0 // 256) + 1) <= 9:
            break
        zoom -= 1
    tiles = {}
    result = np.zeros((grid_size, grid_size), dtype=np.float32)
    with httpx.Client(headers=HEADERS, timeout=12) as client:
        for zi, lat in enumerate(np.linspace(n, s, grid_size)):
            for xi, lon in enumerate(np.linspace(w, e, grid_size)):
                px, py = pixel(float(lat), float(lon))
                tx, ty = int(px // 256), int(py // 256)
                if (tx, ty) not in tiles:
                    path = cache_dir / f"terrain-{zoom}-{tx}-{ty}.png"
                    if path.exists():
                        raw = path.read_bytes()
                    else:
                        response = client.get(f"https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{zoom}/{tx}/{ty}.png")
                        response.raise_for_status()
                        raw = response.content
                    with Image.open(io.BytesIO(raw)) as tile:
                        if tile.size != (256, 256):
                            raise ValueError("Unexpected terrain tile size")
                        tiles[(tx, ty)] = np.asarray(tile.convert("RGB"), dtype=np.float32)
                    if not path.exists():
                        path.write_bytes(raw)
                r, g, b = tiles[(tx, ty)][int(py) % 256, int(px) % 256]
                result[zi, xi] = r * 256 + g + b / 256 - 32768
    minimum = float(result.min())
    return {"width": grid_size, "depth": grid_size, "heights": (result - minimum).round(2).tolist(),
            "minimum_elevation_m": minimum, "relief_m": float(result.max() - minimum),
            "source": "Mapzen/AWS Terrain Tiles (see terrain attribution in report)", "zoom": zoom}


def demo_osm(bbox=None):
    """Original synthetic fixture, geographically anchored only for picker convenience."""
    bbox = bbox or centered_bbox(*DEMO_CENTER)
    w, s, e, n = bbox
    elements = []

    def point(x, z):
        return {"lon": w + (e - w) * x / 250, "lat": n - (n - s) * z / 250}

    def way(coords, tags):
        elements.append({"type": "way", "id": 1000 + len(elements), "tags": tags,
                         "geometry": [point(*p) for p in coords]})

    def rectangle(x, z, width, depth, tags):
        way([(x, z), (x + width, z), (x + width, z + depth), (x, z + depth), (x, z)], tags)

    way([(0, 119), (250, 119)], {"highway": "residential", "width": "10", "name": "Forge Avenue"})
    way([(115, 0), (115, 250)], {"highway": "residential", "width": "9", "name": "Foundry Street"})
    way([(0, 216), (250, 216)], {"highway": "service", "width": "6"})
    for index, (x, z, width, depth, levels) in enumerate([
        (20, 20, 25, 30, 2), (61, 22, 28, 25, 3), (20, 66, 30, 32, 2), (69, 67, 25, 29, 2),
        (140, 19, 32, 29, 3), (192, 21, 36, 30, 4), (144, 70, 35, 28, 2), (197, 73, 29, 25, 2),
        (18, 145, 33, 29, 2), (70, 145, 24, 43, 4), (18, 188, 32, 16, 1),
        (143, 230, 28, 15, 1), (193, 230, 32, 15, 2), (18, 230, 30, 16, 2), (68, 230, 26, 16, 2)]):
        rectangle(x, z, width, depth, {"building": "house" if levels < 3 else "apartments", "building:levels": str(levels),
            "building:material": ["brick", "stone", "wood"][index % 3],
            "roof:shape": "gabled" if levels < 3 else "flat", "name": f"Demo building {index + 1}"})
    rectangle(136, 139, 98, 65, {"leisure": "park", "name": "Forge Gardens"})
    way([(188, 158), (213, 151), (227, 167), (218, 184), (192, 180), (188, 158)], {"natural": "water"})
    for x, z in [(10, 12), (53, 59), (101, 22), (101, 72), (130, 32), (182, 11), (235, 62),
                 (9, 133), (59, 192), (102, 199), (145, 151), (160, 165), (149, 188), (176, 194),
                 (228, 194), (244, 145), (245, 225), (58, 239), (178, 235)]:
        elements.append({"type": "node", "id": 1000 + len(elements), "tags": {"natural": "tree"}, **point(x, z)})
    return {"elements": elements, "realityforge_source": "Original synthetic demonstration fixture; not an actual Toronto reconstruction"}
